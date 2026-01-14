# ---- base ----
FROM python:3.11-slim

# 로그가 바로 찍히도록 + 파이썬 캐시 최소화
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (선택) 빌드에 필요한 기본 패키지들
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# ---- deps ----
# requirements.txt 먼저 복사해서 캐시 활용
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- app ----
# 앱 코드 복사 (레포에 app/ 폴더가 있으니 그대로)
COPY app /app/app

# 컨테이너 외부에서 접근 가능하게
EXPOSE 8000

# ✅ 여기서 "app.main:app" 는 네 FastAPI 엔트리포인트에 맞춰야 해
#   예: app/main.py 안에 app = FastAPI() 가 있으면 아래 그대로 OK
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]