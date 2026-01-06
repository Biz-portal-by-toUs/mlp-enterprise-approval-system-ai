from typing import Iterable, List

from app.schemas import ChatbotRunRequest, ChatHistoryMessage
from app.services.agent_planner import plan_query
from app.services.agent_synthesizer import stream_final_answer
from app.services.agent_tools import execute_rdb_tasks, format_rows
from app.services.callback_client import post_with_retry, validate_callback_url
from app.services.weaviate_store import search_prov_chunks
from app.services.rdb_service import query_db_with_llm


def _history_to_text(history: List[ChatHistoryMessage] | None) -> str:
    if not history:
        return ""
    lines: List[str] = []
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


def _run_rag_tasks(rag_tasks, question: str) -> List[str]:
    contexts: List[str] = []
    tasks = rag_tasks or []
    if not tasks:
        tasks = [{"query": question, "top_k": 5}]
    for t in tasks:
        q = t.get("query") if isinstance(t, dict) else getattr(t, "query", question)
        top_k = t.get("top_k") if isinstance(t, dict) else getattr(t, "top_k", 5)
        try:
            res = search_prov_chunks(q, top_k=top_k)
            contexts.extend(res)
        except Exception as e:
            print(f"[RAG] search failed for {q}: {e}")
    return contexts


def run_chatbot(req: ChatbotRunRequest):
    callback_url = validate_callback_url(req.callbackUrl)
    try:
        hist_preview = _history_to_text(req.history)
        print(f"[CHATBOT] history preview:\n{hist_preview}" if hist_preview else "[CHATBOT] no history provided")

        plan = plan_query(req.question, req.history, req.empId, req.comId)
        print(f"[CHATBOT] plan mode={plan.mode} rag_tasks={len(plan.rag_tasks)} rdb_tasks={len(plan.rdb_tasks)}")

        db_rows: List[dict] = []
        rag_contexts: List[str] = []

        if plan.mode in {"rdb", "hybrid"}:
            rdb_tasks = [t.dict() for t in plan.rdb_tasks]
            rows, summaries = execute_rdb_tasks(rdb_tasks, req.comId, req.empId)
            db_rows.extend(rows)
            print(f"[CHATBOT] RDB tasks summaries: {summaries}")
            if not rows and plan.mode == "rdb":
                # 자유 질의 폴백 (화이트리스트/필터 적용됨)
                try:
                    db_rows = query_db_with_llm(req.question, req.comId, req.empId)
                except Exception as e:
                    print(f"[CHATBOT] fallback LLM SQL failed: {e}")

        if plan.mode in {"rag", "hybrid"}:
            rag_contexts.extend(_run_rag_tasks([t.dict() for t in plan.rag_tasks], req.question))

        # 폴백: 둘 다 비었으면 기본 검색/질의 시도
        if not db_rows and plan.mode == "hybrid":
            try:
                db_rows = query_db_with_llm(req.question, req.comId, req.empId)
            except Exception as e:
                print(f"[CHATBOT] hybrid fallback SQL failed: {e}")
        if not rag_contexts and plan.mode == "hybrid":
            rag_contexts = _run_rag_tasks([], req.question)

        db_text = format_rows(db_rows)
        rag_text = "\n".join(rag_contexts)

        if not db_text and not rag_text:
            msg = "근거와 데이터가 부족해 답변할 수 없습니다.\n"
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "chunk": msg, "done": False, "success": True})
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "done": True, "success": True})
            return

        stream: Iterable[str] = stream_final_answer(
            question=req.question,
            history=req.history,
            db_text=db_text,
            rag_text=rag_text,
            answer_style=plan.answer_style,
            mode=plan.mode,
        )

        for delta in stream:
            payload = {
                "messageId": req.messageId,
                "chunk": delta,
                "done": False,
                "success": True,
            }
            post_with_retry(callback_url, req.callbackKey, payload)

        done_payload = {"messageId": req.messageId, "done": True, "success": True}
        post_with_retry(callback_url, req.callbackKey, done_payload)
    except Exception as e:
        err_msg = str(e)
        print(f"[CHATBOT] error: {err_msg}")
        try:
            error_payload = {
                "messageId": req.messageId,
                "success": False,
                "errorMessage": err_msg,
                "done": True,
            }
            post_with_retry(callback_url, req.callbackKey, error_payload)
        except Exception as cb_err:
            print(f"[CHATBOT] callback failed after error: {cb_err}")
