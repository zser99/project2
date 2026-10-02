"""
[Day1 → Day3] 예측 API  —  serving_app/routers/predict.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   외부 요청을 받아 모델에게 전달하고, 결과를 돌려주는 "창구"입니다.
   계산은 직접 하지 않고, 모델(model_loader)과 감시 도구(retrain_trigger)에게 맡깁니다.

■ 엔드포인트
   [Day1] POST /predict             : 20일치 데이터 + 현재 재고 → 다음날 판매량 예측 + 권장 발주량 (기획서 5-2절)
   [Day3] POST /predict/batch-test  : 긴 판매수량 목록 → 여러 번 예측 → 드리프트 검사

■ 이 파일의 빈칸 : [빈칸 6]  (batch_test 의 슬라이딩 윈도우)
"""
import logging
import math
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException

from data.features import SEQ_LEN  # = 20
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE, compute_rmse
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()
logger = logging.getLogger("aiops")

# 안전재고율: 예측 판매량의 10%를 안전재고로 더 발주합니다. (기획서 5-2절 safety_stock)
SAFETY_STOCK_RATE = 0.10

# (Day3) 최근 예측 기록을 모아 두는 목록.  예: [{"predicted": 61.2, "actual": 63.0}, ...]
#        드리프트 판단은 "최근 21건"(drift_detector.py 의 WINDOW_SIZE)만 보므로 21개까지만 유지합니다.
recent_predictions: list[dict] = []

# (Day3) 시뮬레이션은 판매수량만 보내므로, 이벤트 플래그는 평상시(0)로 고정해서 채웁니다.
SIMULATED_EVENT_FLAG = 0


def _get_model_or_503() -> model_loader.LoadedModel:
    """모델(또는 스케일러)을 불러오지 못하면 500 대신 503 으로 응답합니다. (기획서 5-3절 "모델 로딩 실패")"""
    try:
        return model_loader.get_model()
    except Exception as e:
        logger.error(f"[ERROR] model load failed: {e}")
        raise HTTPException(status_code=503, detail={"status": "error", "message": "운영 모델을 불러오지 못했습니다."})


def _round_half_up(x: float) -> int:
    # 파이썬 round() 는 0.5 를 짝수 쪽으로 보내므로(round(46.5) == 46) 일반적인 반올림을 직접 계산합니다.
    return math.floor(x + 0.5)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    [Day1] 다음날 판매량 예측 + 권장 발주량 계산 (기획서 5-2절)
    받는 것  : {"sequence": [{"sales_qty": 62.0, "event_flag": 0}, ... 20개], "current_stock": 9, "prediction_date": "2026-01-02"}
               20개가 아니거나 값이 음수이거나 current_stock 이 없으면 schemas.py 가 422 에러를 돌려줍니다.
    돌려줄 것: {"prediction_date": "2026-01-02", "predicted_sales": 48, "safety_stock": 5, "current_stock": 9,
               "recommended_order": 44, "model_version": "production-v3", "status": "success"}

    흐름: 모델 가져오기(get_model) → dict 목록으로 변환 → predict_one → 발주량 계산 → 응답 포장
    핵심 계산은 모두 model_loader.predict_one() 안에 있습니다. ([빈칸 2], [빈칸 3])

    권장 발주량 = 예측 판매량 + 안전재고 - 현재 재고   (0 미만이면 0)
      예) 예측 48개, 안전재고 ceil(48 x 0.1) = 5개, 현재 재고 9개  →  48 + 5 - 9 = 44개
    """
    model = _get_model_or_503()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_sales = max(0, _round_half_up(model.predict_one(sequence)))  # 판매량은 음수가 될 수 없음
    # 48 * 0.1 = 4.800000000000001 처럼 소수 오차가 생기므로, 올림 전에 반올림으로 오차를 지웁니다.
    safety_stock = math.ceil(round(predicted_sales * SAFETY_STOCK_RATE, 6))
    recommended_order = max(0, predicted_sales + safety_stock - req.current_stock)
    return PredictResponse(
        prediction_date=req.prediction_date or date.today() + timedelta(days=1),
        predicted_sales=predicted_sales,
        safety_stock=safety_stock,
        current_stock=req.current_stock,
        recommended_order=recommended_order,
        model_version=model.version,
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    [Day3] 드리프트 시뮬레이션
    받는 것  : {"sales": [62.0, 64.1, ... 41개]}   (scripts/simulate_drift.py 가 보냄)
    돌려줄 것: {"predictions": [예측값 21개], "drift_check": {"status": "ok"} 또는 재학습 결과,
               "rmse": 6.4, "threshold": 10.0, "status": "ok", "message": "...", "model_version": "production-v1"}

    ■ 핵심 아이디어: 슬라이딩 윈도우 (20칸짜리 창문을 한 칸씩 밀기)
      판매수량 41개가 들어오면, 20개씩 잘라 "그다음 날"을 예측하고 실제 값과 비교합니다.

        i=0 : [p0  ~ p19] → 예측   vs  실제 p20
        i=1 : [p1  ~ p20] → 예측   vs  실제 p21
        ...
        i=20: [p20 ~ p39] → 예측   vs  실제 p40
        → 총 41 - 20 = 21번 예측 = 드리프트 판단에 필요한 21건이 딱 채워집니다.

    확인 방법
      python scripts/simulate_drift.py
        [normal]          drift_check = {'status': 'ok'}
        [drift_injection] drift_check = {'status': 'retrain_triggered', 'promoted': True, ...}
      /docs 에서 직접 호출할 때는 predictions 가 (판매수량 개수 - 20)개인지 확인하세요.
    """
    model = _get_model_or_503()
    predictions: list[float] = []

    sales = req.sales
    for i in range(len(sales) - SEQ_LEN):
        # ════════════════════════════ [빈칸 6] ════════════════════════════
        # i번째 창문(window)의 시작·끝 위치와, 그 창문 바로 다음 날(actual)의 위치를 채우세요. (i 와 SEQ_LEN 으로)
        #   (위 docstring 의 그림에서 i=0 일 때 무엇이 창문이고 무엇이 실제 값인지 먼저 확인)
        #
        #   생각해 볼 질문
        #     · 파이썬 슬라이싱 sales[a:b] 는 b 를 포함하나요?
        #     · 실제 값을 한 칸 앞(창문의 마지막 날)으로 잡으면, 모델은 무엇을 "맞힌" 셈이 될까요?
        #     · 반대로 창문을 한 칸 더 길게 잡아서 실제 값이 창문 안에 들어가면 RMSE는 어떻게 될까요?
        window = sales[i : i+SEQ_LEN]                                                        ######### 해낸다.. 제발 맞아라... 최종으로 돌렸을때 RMSE가 생각보다 너무 정확하면 원인은 이놈임.
        sequence = [{"sales_qty": q, "event_flag": SIMULATED_EVENT_FLAG} for q in window]
        pred = model.predict_one(sequence)
        actual = sales[i+SEQ_LEN]                                                            ########## 이것도!!!
        predictions.append(pred)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 21건만 남기기 — 오래된 기록까지 섞이면 "지금" 상태를 판단할 수 없습니다.
    # (recent_predictions = ... 로 쓰면 함수 안의 새 변수가 되므로, [:] 로 목록 내용을 바꿉니다)
    recent_predictions[:] = recent_predictions[-21:]  # WINDOW_SIZE 유지

    # 드리프트 판단·재학습은 retrain_trigger.py 가 합니다. 여기서는 넘겨주기만!
    drift_check = check_and_trigger(recent_predictions)

    # 새 버전이 Production 으로 승격됐으면 서빙 중인 모델도 교체합니다.
    # (get_model 은 한 번 불러온 모델을 계속 재사용하므로, 이걸 안 하면 다음 /predict 도 옛 모델이 응답합니다)
    if drift_check.get("promoted"):
        model_loader.reload_model()

    serving_version = _get_model_or_503().version
    return BatchTestResponse(
        predictions=predictions,
        drift_check=drift_check,
        rmse=round(compute_rmse(recent_predictions[-WINDOW_SIZE:]), 2),
        threshold=RMSE_THRESHOLD,
        status=drift_check["status"],
        message=_batch_message(drift_check, len(recent_predictions), serving_version),
        model_version=serving_version,
    )


def _batch_message(drift_check: dict, n_records: int, serving_version: str) -> str:
    if drift_check.get("status") != "retrain_triggered":
        if n_records < WINDOW_SIZE:
            return f"판정에 필요한 예측 {WINDOW_SIZE}건 중 {n_records}건만 쌓여 아직 드리프트를 판정하지 않았습니다."
        return "예측 오차가 임계값 이내입니다. 현재 모델을 유지합니다."
    if drift_check.get("promoted"):
        # MODEL_SOURCE=local 이거나 교체에 실패하면 서빙 모델은 그대로이므로, "교체했다"고 단정하지 않고 현재 버전을 알려줍니다.
        return f"예측 오차가 임계값을 초과하여 재학습했고, 새 모델이 Production 으로 승격됐습니다. (현재 서빙 모델: {serving_version})"
    return "예측 오차가 임계값을 초과하여 재학습했지만, 새 모델이 배포 기준을 통과하지 못해 기존 모델을 유지합니다."
