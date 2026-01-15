from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status, Body

import weaviate
from weaviate.connect import ConnectionParams

from app.core.config import settings
from app.schemas import (
    ProvEmbeddingDeleteRequest,
    ProvEmbeddingRequest,
    ProvEmbeddingStatusUpdateRequest,
    RunRequest,
)
from app.routers.chatbot import router as chatbot_router
from app.workers.meetings import process_job
from app.workers.prov_documents import process_prov_embedding
from app.services.provdocuments.weaviate_store import delete_prov_chunks, update_prov_chunks_public

app = FastAPI(title="Meeting AI")
app.include_router(chatbot_router)


# client = weaviate.WeaviateClient(
#     connection_params=ConnectionParams.from_url(
#         "http://localhost:8080",
#         grpc_port=50051,
#     )
# )
# client.connect()

# print(client.get_meta())  # {"version": "..."} 등 출력되면 연결 OK

# client.close()

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ai/meetings/run")
def run_ai(req: RunRequest, background: BackgroundTasks):
    print(f"[AI RUN] meetNo={req.meetNo}, title={req.meetingTitle!r}")
    """
    Spring -> FastAPI 호출용.
    즉시 200 반환하고, 백그라운드에서 처리 후 callbackUrl로 결과 전송.
    """
    background.add_task(process_job, req)
    return {"queued": True, "meetNo": req.meetNo}


@app.post("/api/v1/prov-documents/embedding")
def run_prov_embedding(req: ProvEmbeddingRequest, background: BackgroundTasks):
    print(f"[PROV EMBEDDING RUN] provNo={req.provNo}, objectKey={req.objectKey}")
    """
    규정 문서 임베딩 요청 (Spring -> FastAPI).
    즉시 200 반환 후 비동기로 S3 다운로드 + 텍스트 추출 + 임베딩 처리.
    """
    background.add_task(process_prov_embedding, req)
    return {"queued": True, "provNo": req.provNo}


@app.delete("/api/v1/prov-documents/embedding")
def delete_prov_embedding(
    req: ProvEmbeddingDeleteRequest = Body(...), 
    x_callback_secret: str = Header(..., alias="X-CALLBACK-SECRET", convert_underscores=False),
):
    expected = settings.CALLBACK_KEY
    if not expected or x_callback_secret != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    deleted = delete_prov_chunks(req.comId, req.provNo)
    return {"deleted": deleted, "comId": req.comId, "provNo": req.provNo}


@app.patch("/api/v1/prov-documents/embedding/status")
def update_prov_embedding_status(
    req: ProvEmbeddingStatusUpdateRequest = Body(...),
    x_callback_secret: str = Header(..., alias="X-CALLBACK-SECRET", convert_underscores=False),
):
    expected = settings.CALLBACK_KEY
    if not expected or x_callback_secret != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    updated = update_prov_chunks_public(req.comId, req.provNo, req.isPublic)
    return {
        "updated": updated,
        "comId": req.comId,
        "provNo": req.provNo,
        "isPublic": req.isPublic,
    }
