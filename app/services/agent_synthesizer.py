from typing import Iterable, List

from app.clients import openai_client
from app.core.config import settings
from app.schemas import ChatHistoryMessage


def _history_to_text(history: List[ChatHistoryMessage] | None) -> str:
    if not history:
        return ""
    lines = []
    for m in history:
        if isinstance(m, dict):
            role = str(m.get("role", "")).lower()
            content = m.get("content", "")
        else:
            role = m.role.lower()
            content = m.content
        prefix = "User" if role.startswith("user") else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def stream_final_answer(
    question: str,
    history: List[ChatHistoryMessage] | None,
    db_text: str,
    rag_text: str,
    answer_style: str | None,
    mode: str,
) -> Iterable[str]:
    """
    DB와 RAG 근거를 모두 사용해 최종 답변을 스트리밍한다.
    """
    history_text = _history_to_text(history)
    style_hint = f"답변 스타일: {answer_style}" if answer_style else ""
    user_prompt = (
        "아래 DB 결과와 규정 근거를 활용해 한국어로 간결하고 정확하게 답변하세요. "
        "DB는 사실 데이터, RAG는 규정/정책 근거입니다. 정보가 없으면 모른다고 말하세요. "
        f"{style_hint}\n\n"
        f"[이전 대화]\n{history_text}\n\n" if history_text else ""
        f"[질문]\n{question}\n\n"
        f"[DB 결과]\n{db_text or '(없음)'}\n\n"
        f"[규정 근거]\n{rag_text or '(없음)'}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=settings.SUM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 사내 전자결재/그룹웨어 챗봇이다. DB 결과는 사실, 규정 근거는 정책이다. "
                        "출처가 없는 내용은 추측하지 말고, 필요시 근거/데이터 부족을 명시한다."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        resp = openai_client.chat.completions.create(
            model=settings.SUM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 사내 전자결재/그룹웨어 챗봇이다. DB 결과는 사실, 규정 근거는 정책이다. "
                        "출처가 없는 내용은 추측하지 말고, 필요시 근거/데이터 부족을 명시한다."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        full = resp.choices[0].message.content or ""
        for para in full.split("\n\n"):
            part = para.strip()
            if part:
                yield part + "\n"
