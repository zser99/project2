"""
Day1: FastAPI 요청/응답 Pydantic 스키마.

LSTM은 한 시점의 값이 아니라 최근 SEQ_LEN(20)일의 흐름을 입력받아야 하므로,
/predict는 단일 행이 아니라 "20일치 시퀀스"를 요청 본문으로 받습니다.
이 검증 로직은 Day2 "데이터/모델 검증" 실습에서 다루는 것과 같은 종류입니다 -
서빙 시점 입력 검증이 학습 시점 피처(data/features.py)와 어긋나지 않도록
길이(SEQ_LEN)와 값 범위(ge=0, le=1)를 스키마 단에서 강제합니다.
검증에 실패하면 FastAPI가 422를 돌려주고, 메시지는 serving_app/errors.py가 한국어로 바꿉니다.

/predict 요청·응답 형식은 기획서 5-2절(요청/응답 예시)을 따릅니다.
"""
from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from data.features import SEQ_LEN

# Swagger(/docs)의 "Try it out" 기본값 - data/sample_fresh_sales.csv 의 마지막 20일(2025-01-18 ~ 2025-02-06)
_EXAMPLE_SEQUENCE: list[JsonValue] = [
    {"sales_qty": q, "event_flag": e}
    for q, e in [
        (60, 1), (60, 1), (95, 1), (68, 0), (67, 0), (62, 0), (62, 0), (44, 0), (41, 0), (65, 0),
        (51, 0), (51, 0), (48, 0), (63, 0), (41, 0), (38, 0), (66, 0), (63, 0), (63, 0), (69, 0),
    ]
]


class DailyPoint(BaseModel):
    sales_qty: float = Field(..., ge=0, description="해당 날짜 판매수량 (0 이상)")
    event_flag: int = Field(..., ge=0, le=1, description="해당 날짜 이벤트 여부 (0 또는 1)")


class PredictRequest(BaseModel):
    sequence: list[DailyPoint] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 날 -> 가장 최근 날 순서의 최근 {SEQ_LEN}일 시퀀스",
    )
    current_stock: int = Field(..., ge=0, description="발주 시점의 현재 재고 (권장 발주량 계산에 사용)")
    prediction_date: date | None = Field(None, description="예측 대상 날짜 (생략하면 오늘 기준 다음날)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"sequence": _EXAMPLE_SEQUENCE, "current_stock": 9, "prediction_date": "2025-02-07"}]
        }
    )


class PredictResponse(BaseModel):
    prediction_date: date = Field(..., description="예측 대상 날짜")
    predicted_sales: int = Field(..., description="모델이 예측한 다음날 판매량 (개, 반올림)")
    safety_stock: int = Field(..., description="안전재고 = 예측 판매량 x 안전재고율(10%), 올림")
    current_stock: int = Field(..., description="발주 시점의 현재 재고 (요청값 그대로)")
    recommended_order: int = Field(..., description="권장 발주량 = 예측 판매량 + 안전재고 - 현재 재고 (0 미만이면 0)")
    model_version: str = Field(..., description="예측에 사용한 모델 버전 (예: production-v3, v1-local)")
    status: str = Field(default="success", description="처리 결과")


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션에서 사용 (scripts/simulate_drift.py 참고)
    # SEQ_LEN + N 개의 연속된 판매수량을 보내면, 서버가 내부적으로 슬라이딩 윈도우로 잘라
    # 여러 건을 연속 예측한다. (이벤트 플래그는 시뮬레이션이므로 평상시(0)로 고정)
    sales: list[Annotated[float, Field(ge=0)]] = Field(
        ...,
        min_length=SEQ_LEN + 1,
        description=f"연속된 일별 판매수량 (최소 {SEQ_LEN + 1}개). 41개를 보내면 21번 예측해 드리프트를 바로 판정합니다.",
    )


class BatchTestResponse(BaseModel):
    # predictions·drift_check 는 대시보드(static/index.html)와 scripts/simulate_drift.py 가 읽으므로 형식을 유지하고,
    # 기획서 5-2절의 rmse·threshold·status·message 를 함께 돌려준다.
    predictions: list[float] = Field(..., description="슬라이딩 윈도우마다의 다음날 판매량 예측값")
    drift_check: dict = Field(..., description='드리프트 판정·재학습 결과 (예: {"status": "ok"})')
    rmse: float = Field(..., description="최근 판정 윈도우(최대 21건)의 예측 RMSE (개)")
    threshold: float = Field(..., description="드리프트 판정 기준 RMSE (monitoring/drift_detector.py 의 RMSE_THRESHOLD)")
    status: str = Field(..., description='"ok" 또는 "retrain_triggered"')
    message: str = Field(..., description="판정 결과 설명")
    model_version: str = Field(..., description="응답 시점에 서빙 중인 모델 버전 (재학습으로 승격되면 새 버전)")
