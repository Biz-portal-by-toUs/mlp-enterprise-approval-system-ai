import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List
from app.core.config import settings

import boto3
import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Meeting AI")

# ✅ settings 객체의 API 키 사용
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ====== Global Settings (settings 객체 사용) ======
SPLIT_SECONDS = settings.SPLIT_SECONDS
STT_MODEL = settings.STT_MODEL
SUM_MODEL = settings.SUM_MODEL
CALLBACK_HEADER = settings.CALLBACK_HEADER

# ====== ENV ======
SPLIT_SECONDS = settings.SPLIT_SECONDS   # 10분=600
STT_MODEL = settings.STT_MODEL
SUM_MODEL = settings.SUM_MODEL

# Spring 콜백 검사용 헤더
CALLBACK_HEADER = settings.CALLBACK_HEADER

# (fallback 용) FastAPI에서 presigned URL 만들 수 있게만 남겨둠
AWS_REGION = settings.AWS_REGION
AWS_BUCKET = settings.AWS_BUCKET
PRESIGN_EXPIRE = settings.PRESIGN_EXPIRE
s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name=settings.AWS_REGION
)


# ====== DTO ======
class RunRequest(BaseModel):
    meetNo: int
    objectKey: str

    # ✅ Spring이 넘겨주면 FastAPI는 S3 직접 접근 안 함 (권장)
    downloadUrl: Optional[str] = None

    # ✅ Spring 콜백 (결과 저장은 Spring이 함)
    callbackUrl: str
    callbackKey: str

    # 옵션
    sttModel: Optional[str] = None
    summaryModel: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ai/meetings/run")
def run_ai(req: RunRequest, background: BackgroundTasks):
    """
    Spring -> FastAPI 호출용.
    즉시 200 반환하고, 백그라운드에서 처리 후 callbackUrl로 결과 전송.
    """
    background.add_task(process_job, req)
    return {"queued": True, "meetNo": req.meetNo}


# ====== Utils ======
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError("ffmpeg가 필요합니다. 서버에 ffmpeg 설치해줘요.") from e


def split_audio(input_path: Path, out_dir: Path, seconds: int) -> List[Path]:
    """
    webm/opus 같은 입력을 N초 단위 mp3로 분할 (재인코딩 방식)
    """
    ensure_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%03d.mp3")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(seconds),
        "-reset_timestamps", "1",
        "-c:a", "libmp3lame",
        "-q:a", "4",
        pattern
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("오디오 분할 결과가 없습니다.")
    return chunks


def whisper_transcribe(file_path: Path, model_name: str) -> str:
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=model_name,
            file=f,
        )
    return tr.text or ""


def gpt_summarize(transcribed_text: str, model_name: str) -> str:
    detailed_prompt = f"""
당신은 꼼꼼하고 전문적인 회의록 작성 서기입니다.
아래 제공된 회의 녹취록(Transcript)을 바탕으로, 빠진 내용 없이 상세한 회의록을 작성해 주세요.

반드시 다음 형식을 지켜서 작성해 주세요:

## 1. 회의 개요
- 회의의 전반적인 주제와 목적을 한 문단으로 요약

## 2. 주요 안건 및 논의 내용 (상세)
- 회의에서 논의된 각 안건별로 누가 어떤 의견을 냈는지 구체적으로 서술
- 중요한 숫자가 나왔다면 반드시 포함
- 찬성/반대 의견이 갈렸다면 양쪽 입장을 모두 정리

## 3. 결정 사항
- 회의를 통해 확정된 내용을 명확하게 나열

## 4. 향후 행동 계획 (Action Items)
- [담당자] 할 일 (기한) 형식으로 정리 (담당자가 명확하지 않으면 '미정'으로 표시)

---
[녹취록 전문]
{transcribed_text}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a professional meeting minutes assistant. You create detailed, structured reports in Korean."},
            {"role": "user", "content": detailed_prompt},
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content or ""


def format_callback_url(raw: str, meet_no: int) -> str:
    if "{meetNo}" in raw:
        return raw.replace("{meetNo}", str(meet_no))
    return raw


def presign_get_url(object_key: str) -> str:
    """
    ✅ settings에서 bucket과 expire 시간을 가져옴
    """
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.AWS_BUCKET, "Key": object_key},
        ExpiresIn=settings.PRESIGN_EXPIRE
    )


def download_audio(download_url: str, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300) as http:
        r = http.get(download_url)
        r.raise_for_status()
        dst_path.write_bytes(r.content)


def callback_to_spring(callback_url: str, callback_key: str, payload: dict):
    headers = {CALLBACK_HEADER: callback_key, "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as http:
        r = http.patch(callback_url, headers=headers, json=payload)  # ✅ PATCH로 변경
        print("✅ CALLBACK REQ URL:", callback_url)
        print("✅ CALLBACK RES:", r.status_code, r.text)  # <- 이게 핵심
        r.raise_for_status()


# ====== Worker ======
def process_job(req: RunRequest):
    print("=== JOB START ===", req.meetNo, req.objectKey)
    meet_no = req.meetNo
    object_key = req.objectKey

    stt_model = req.sttModel or settings.STT_MODEL
    sum_model = req.summaryModel or settings.SUM_MODEL

    cb_url = format_callback_url(req.callbackUrl, meet_no)

    try:
        with tempfile.TemporaryDirectory() as td:
            print("STEP1: downloading...")
            td = Path(td)
            audio_path = td / "input.webm"
            chunks_dir = td / "chunks"

            # 1) ✅ 오디오 다운로드 (Spring이 downloadUrl 주면 그거 사용)
            download_url = req.downloadUrl or presign_get_url(object_key)
            download_audio(download_url, audio_path)
            print("STEP1 DONE bytes=", audio_path.stat().st_size)

            # 2) 분할
            print("STEP2: splitting...")
            chunks = split_audio(audio_path, chunks_dir, SPLIT_SECONDS)
            print("STEP2 DONE chunks=", len(chunks))

            # 3) Whisper STT
            print("STEP3: whisper...")
            texts = []
            for chunk in chunks:
                t = whisper_transcribe(chunk, stt_model)
                texts.append(t)

            transcribed_text = "\n\n".join(texts).strip()
            print("STEP3 DONE stt_len=", len(transcribed_text))

            # 4) GPT 요약
            
            print("STEP4: gpt summarize...")
            summary = gpt_summarize(transcribed_text, sum_model)
            print("STEP4 DONE ai_len=", len(summary))
            

            # 5) ✅ Spring 콜백 (applyAiResult DTO에 맞춤)
            payload = {
                "meetNo": meet_no,            # ✅ 필수
                "objectKey": object_key,      # (DTO에 있으면 같이)
                "status": "DONE",
                "sttText": transcribed_text,
                "aiText": summary,
                "errorMessage": None
            }
            callback_to_spring(cb_url, req.callbackKey, payload)

    except Exception as e:
        print("=== JOB FAIL ===", repr(e))
        payload = {
            "meetNo": meet_no,            # ✅ 필수
            "objectKey": object_key,
            "status": "FAILED",
            "sttText": None,
            "aiText": None,
            "errorMessage": str(e)
        }
        try:
            callback_to_spring(cb_url, req.callbackKey, payload)
        except Exception:
            pass