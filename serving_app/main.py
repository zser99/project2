"""
FastAPI 앱 진입점.

Day1: app 생성, 라우터(predict, health) 등록, 시작 시(lifespan) 로딩 모드에 따라 모델 준비
Day2: data 라우터 등록 (판매 데이터 업로드)
Day3: "aiops" 로거를 logs/aiops.log 파일로 연결(로깅 설정) + logs 라우터(로그 파일 조회) 등록

운영 설계(기획서 3-2절)에 맞춰 추가한 것
  · 422 메시지 한국어화 : serving_app/errors.py
  · 응답 시간 측정     : 모든 응답에 X-Process-Time 헤더, /predict 가 1초를 넘으면 aiops.log 에 [WARN]

정적 대시보드: serving_app/static/index.html 이 /health · /predict · /predict/batch-test ·
/data/upload · /logs 를 호출하는 확인용 화면입니다. API 라우터를 먼저 등록한 뒤
StaticFiles를 "/"에 마지막으로 mount해야, /predict 같은 API 경로가 정적 파일보다
먼저 매칭됩니다(Starlette는 등록 순서대로 라우트를 검사합니다).
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.errors import register_error_handlers
from serving_app.routers import data, health, logs, predict

# monitoring/retrain_trigger.py가 쓰는 "aiops" 로거를 logs/aiops.log 파일에 연결한다.
# (routers/logs.py가 같은 디렉토리를 읽기 전용으로 노출한다.) 여기서 이 로거 하나만
# 직접 설정하므로, uvicorn 자체 로깅 설정과 충돌하지 않는다.
_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(os.path.join(_LOG_DIR, "aiops.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())  # 터미널에서도 동일하게 확인 가능

# 예측 응답 시간 목표 (기획서 2-3절 "예측 응답 시간 1초 이내", 3-2절 "1초 초과 시 경고")
LATENCY_SLO_SECONDS = 1.0
LATENCY_SLO_PATHS = {"/predict"}  # batch-test 는 재학습(수 초)을 포함하므로 대상에서 뺀다


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Day1 실습 포인트: LOADING_MODE=eager 로 켜고 서버 시작 시간을 lazy와 비교해보세요.
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        try:
            model_loader.load_eager()
        except Exception as e:
            # 모델을 못 불러와도 서버는 띄워 둔다: /health 로 상태를 확인할 수 있고, /predict 는 503 을 돌려주며
            # 다음 요청 때 다시 불러오기를 시도한다 (그 사이 train_and_register.py 로 모델을 만들면 바로 복구됨).
            _aiops_logger.error(f"[ERROR] eager model load failed - serving starts without a model: {e}")
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
    yield


app = FastAPI(title="Fresh Sales Serving & AIOps", lifespan=lifespan)
register_error_handlers(app)


@app.middleware("http")
async def measure_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    if request.url.path in LATENCY_SLO_PATHS and elapsed > LATENCY_SLO_SECONDS:
        _aiops_logger.warning(
            f"[WARN] slow response - {request.method} {request.url.path} {elapsed:.2f}s > {LATENCY_SLO_SECONDS:.2f}s"
        )
    return response


app.include_router(predict.router)
app.include_router(health.router)
app.include_router(data.router)  # 판매 데이터 업로드
app.include_router(logs.router)  # 대시보드: 재학습 로그 파일 조회

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")  # 대시보드 UI
