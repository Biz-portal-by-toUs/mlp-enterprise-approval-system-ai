from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    EMBED_MODEL: str = "text-embedding-3-small"
    EMBED_CHUNK_WORDS: int = 400
    EMBED_CHUNK_OVERLAP: int = 50
    RDB_MODEL: str = "gpt-4o-mini"

    # AI 모델 설정 (기본값 설정 가능)
    SPLIT_SECONDS: int = 600
    STT_MODEL: str = "whisper-1"
    SUM_MODEL: str = "gpt-4o"

    # Spring 콜백 설정
    CALLBACK_HEADER: str 
    CALLBACK_KEY: str 
    CALLBACK_BASE_URL: str | None = None
    CALLBACK_DELETE_HEADER: str = "X-CALLBACK-SECRET"

    # Weaviate
    WEAVIATE_HTTP_URL: str | None = "http://localhost:8080"
    WEAVIATE_GRPC_PORT: int = 50051
    WEAVIATE_COLLECTION: str = "ProvDocuments"

    # RDB (직원 정보 조회 등)
    EMP_DB_DSN: str | None = None  # 예: sqlite:////path/to/file.db 또는 postgres://...

    # AWS S3 설정
    AWS_REGION: str 
    AWS_BUCKET: str 
    PRESIGN_EXPIRE: int = 3600

    # .env 파일 위치 (app/core/config.py 기준 루트 폴더의 .env)
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../.env"),
        env_file_encoding='utf-8',
        extra='ignore' # .env에 클래스 정의 외의 변수가 있어도 무시
    )

settings = Settings()
