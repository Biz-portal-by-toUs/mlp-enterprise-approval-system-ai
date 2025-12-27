from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str

    # AI 모델 설정 (기본값 설정 가능)
    SPLIT_SECONDS: int = 600
    STT_MODEL: str = "whisper-1"
    SUM_MODEL: str = "gpt-4o"

    # Spring 콜백 설정
    CALLBACK_HEADER: str = "X-Internal-Callback-Key"

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