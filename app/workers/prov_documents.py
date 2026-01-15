import tempfile
from pathlib import Path

from app.core.config import settings
from app.schemas import ProvEmbeddingRequest
from app.services.callbacks import callback_to_spring
from app.services.provdocuments.documents import chunk_by_article, download_object, extract_text
from app.services.provdocuments.embeddings import embed_chunks
from app.services.provdocuments.weaviate_store import store_prov_chunks


def _format_callback_url(raw: str, prov_no: int) -> str:
    if "{provNo}" in raw:
        return raw.replace("{provNo}", str(prov_no))
    return raw


def _absolute_callback_url(cb: str) -> str:
    """Allow passing relative path like /api/v1/prov-documents/{provNo}/embedding."""
    if cb.startswith("http://") or cb.startswith("https://"):
        return cb
    base = settings.CALLBACK_BASE_URL
    if base:
        return base.rstrip("/") + "/" + cb.lstrip("/")
    return cb  # fallback: try as-is; will fail loudly if invalid


def process_prov_embedding(req: ProvEmbeddingRequest):
    prov_no = req.provNo
    callback_url = _absolute_callback_url(_format_callback_url(req.callbackUrl, prov_no))
    callback_key = req.callbackKey or settings.CALLBACK_KEY
    if not callback_key:
        raise RuntimeError("callbackKey가 없습니다. 요청에 포함하거나 CALLBACK_KEY env를 설정하세요.")

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            file_path = td_path / req.originalName

            print(f"[PROV] STEP1 download -> {file_path}")
            download_object(req.objectKey, file_path)

            print(f"[PROV] STEP2 extract text")
            text = extract_text(file_path, req.contentType)
            print(f"[PROV] extracted chars={len(text)}")

            base_title = Path(req.originalName).stem or req.originalName
            print(f"[PROV] STEP3 chunking by article docTitle={base_title}")
            doc_title, chunks = chunk_by_article(
                text,
                base_title,
                settings.EMBED_CHUNK_WORDS,
                settings.EMBED_CHUNK_OVERLAP,
            )
            print(f"[PROV] chunk count={len(chunks)} docTitle={doc_title}")

            print(f"[PROV] STEP4 embedding start model={settings.EMBED_MODEL}")
            # 실제 임베딩 (필요시 이 결과를 벡터DB 등에 저장)
            embs = embed_chunks(chunks)
            try:
                print(f"[PROV] embedding done shape={embs.shape} dtype={embs.dtype}")
                # 첫 번째 벡터 일부 샘플(과도한 로그 방지)
                first = embs[0][:8].tolist() if len(embs) else []
                print(f"[PROV] first vector (8 dims) preview={first}")
            except Exception:
                print("[PROV] embedding done (shape 확인 실패)")

            # ✅ 회사별 메타를 포함해 벡터 DB 저장
            try:
                store_prov_chunks(
                    com_id=req.comId,
                    prov_no=prov_no,
                    object_key=req.objectKey,
                    original_name=req.originalName,
                    is_public=req.isPublic,
                    chunks=chunks,
                    embeddings=embs,
                )
                print(f"[PROV] weaviate stored chunks={len(chunks)} collection={settings.WEAVIATE_COLLECTION}")
            except Exception as e:
                print(f"[PROV] weaviate store failed: {e}")
                raise

        payload = {
            "provNo": prov_no,
            "success": True,
            "chunkCnt": len(chunks),
            "errorMsg": None,
        }
        try:
            key_preview = (callback_key[:3] + "***") if callback_key else "(none)"
            print(f"[PROV] CALLBACK header={settings.CALLBACK_HEADER} key={key_preview} url={callback_url}")
            callback_to_spring(callback_url, callback_key, payload)
        except Exception as e:
            print(f"[PROV] callback failed: {e}")
            raise

    except Exception as e:
        payload = {
            "provNo": prov_no,
            "success": False,
            "chunkCnt": None,
            "errorMsg": str(e),
        }
        try:
            callback_to_spring(callback_url, callback_key or "", payload)
        except Exception:
            pass
