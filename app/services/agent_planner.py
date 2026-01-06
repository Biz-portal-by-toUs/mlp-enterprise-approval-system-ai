import json
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError

from app.clients import openai_client
from app.core.config import settings


class RagTask(BaseModel):
    query: str
    top_k: int = 5


class RdbTask(BaseModel):
    name: str
    args: dict = {}


class QueryPlan(BaseModel):
    mode: Literal["rdb", "rag", "hybrid"] = "rag"
    rag_tasks: List[RagTask] = []
    rdb_tasks: List[RdbTask] = []
    answer_style: Optional[str] = None


def _history_to_text(history) -> str:
    if not history:
        return ""
    lines = []
    for m in history:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
        if not content:
            continue
        prefix = "User" if str(role).lower().startswith("user") else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def plan_query(question: str, history, emp_id: str, com_id: Optional[str]) -> QueryPlan:
    """
    LLM 기반 플래너: rdb/rag/hybrid 플랜(JSON)을 생성하고 검증한다.
    """
    history_text = _history_to_text(history)
    user_block = (
        f"[이전 대화]\n{history_text}\n\n[현재 질문]\n{question}"
        if history_text
        else question
    )
    system_prompt = (
        "너는 사내 챗봇 플래너다. 질문을 해결하기 위해 DB 조회(RDB), 규정 검색(RAG), 또는 둘 다(Hybrid) 계획을 JSON으로만 출력한다.\n"
        "- mode: rdb | rag | hybrid\n"
        "- rag_tasks: [{\"query\": \"...\", \"top_k\": 5}]\n"
        "- rdb_tasks: [{\"name\": \"task_name\", \"args\": {...}}]\n"
        "- answer_style: 요약/비교/추천 등 힌트\n"
        "규정/정책/조항 해석은 rag, 직원/회사 데이터/개수/목록/일정/예약/연락처는 rdb, 둘 다 필요하면 hybrid.\n"
        "JSON만 출력하고, 설명은 쓰지 마."
    )
    try:
        resp = openai_client.chat.completions.create(
            model=settings.RDB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_block},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        plan_dict = json.loads(raw)
        return QueryPlan(**plan_dict)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"[PLANNER] parse failed: {e}")
        return QueryPlan(mode="rag", rag_tasks=[RagTask(query=question)])
    except Exception as e:
        print(f"[PLANNER] LLM failed: {e}")
        return QueryPlan(mode="rag", rag_tasks=[RagTask(query=question)])
