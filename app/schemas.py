from typing import Optional

from pydantic import BaseModel


class RunRequest(BaseModel):
    meetNo: int
    objectKey: str
    downloadUrl: Optional[str] = None
    callbackUrl: str
    callbackKey: str
    meetingTitle: Optional[str] = None
    sttModel: Optional[str] = None
    summaryModel: Optional[str] = None


class ProvEmbeddingRequest(BaseModel):
    provNo: int
    comId: str
    objectKey: str
    originalName: str
    contentType: Optional[str] = None
    size: Optional[int] = None
    isPublic: Optional[bool] = None
    callbackUrl: str
    callbackKey: Optional[str] = None


class ProvEmbeddingDeleteRequest(BaseModel):
    comId: str
    provNo: int


class ProvEmbeddingStatusUpdateRequest(BaseModel):
    comId: str
    provNo: int
    isPublic: bool


class ChatbotRunRequest(BaseModel):
    messageId: str
    empId: str
    comId: Optional[str] = None
    question: str
    callbackUrl: str
    callbackKey: str
    sessionId: Optional[str] = None  # Spring에서 action 연동 시 필요할 수 있음
    history: Optional[list["ChatHistoryMessage"]] = None


class ChatHistoryMessage(BaseModel):
    role: str  # user | assistant
    content: str


# Forward reference resolve
ChatbotRunRequest.model_rebuild()
