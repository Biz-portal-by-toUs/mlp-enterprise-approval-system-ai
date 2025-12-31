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
    callbackUrl: str
    callbackKey: Optional[str] = None


class ProvEmbeddingDeleteRequest(BaseModel):
    comId: str
    provNo: int
