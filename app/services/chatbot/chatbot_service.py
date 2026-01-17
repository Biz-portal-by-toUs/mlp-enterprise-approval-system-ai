from typing import Iterable, List

from app.schemas import ChatbotRunRequest
from app.services.chatbot.agent_planner import plan_query
from app.services.chatbot.agent_synthesizer import stream_final_answer
from app.services.chatbot.callback_client import post_with_retry, validate_callback_url
from app.services.provdocuments.weaviate_store import search_prov_chunks
from app.services.chatbot.rdb_service import query_db_with_llm
from app.services.chatbot.agent_tools import format_rows
from app.services.chatbot.utils import _history_to_text


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
            # 자동 Text-to-SQL만 사용 (사전 정의 태스크 미사용)
            try:
                db_rows = query_db_with_llm(req.question, req.comId, req.empId)
                print(f"[CHATBOT] db_rows: {db_rows}")
            except Exception as e:
                import traceback
                print(f"[CHATBOT] LLM SQL failed: {e}\n{traceback.format_exc()}")

        if plan.mode in {"rag", "hybrid"}:
            rag_contexts.extend(_run_rag_tasks([t.model_dump() for t in plan.rag_tasks], req.question))

        db_text = format_rows(db_rows)
        print("[CHATBOT] db_text: "+db_text)
        rag_text = "\n".join(rag_contexts)
        print("[CHATBOT] rag_text: "+rag_text)

        if not db_text and not rag_text:
            msg = "근거와 데이터가 부족해 답변할 수 없습니다.\n"
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "chunk": msg, "done": False, "success": True})
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "done": True, "success": True})
            return

        stream: Iterable[dict] = stream_final_answer(
            question=req.question,
            history=req.history,
            db_text=db_text,
            rag_text=rag_text,
            answer_style=plan.answer_style,
            mode=plan.mode,
        )

        try:
            seq = 0
            full_answer_parts: List[str] = []
            for delta in stream:
                if delta.get("chunk"):
                    chunk = delta["chunk"]
                    full_answer_parts.append(chunk)
                    payload = {
                        "messageId": req.messageId,
                        "chunk": chunk,
                        "seq": seq,
                        "done": False,
                        "success": True,
                    }
                    print(f"[CHATBOT] stream chunk seq={seq} messageId={req.messageId} delta={delta['chunk']!r}")
                    seq += 1
                    post_with_retry(callback_url, req.callbackKey, payload)
                if delta.get("done"):
                    action = delta.get("action")
                    done_payload = {
                        "messageId": req.messageId,
                        "done": True,
                        "success": True,
                        "fullText": "".join(full_answer_parts),
                    }
                    if action:
                        done_payload["actionId"] = action.get("actionId")
                        done_payload["params"] = action.get("params")
                    print(f"[CHATBOT] done payload -> {done_payload}")
                    post_with_retry(callback_url, req.callbackKey, done_payload)
        except Exception as e:
            print(f"[CHATBOT] stream failed: {e}")
            msg = "근거와 데이터가 부족해 답변할 수 없습니다.\n"
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "chunk": msg, "done": False, "success": True})
            post_with_retry(callback_url, req.callbackKey, {"messageId": req.messageId, "done": True, "success": True})
            return
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
