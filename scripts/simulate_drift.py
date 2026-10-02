"""
Day3 드리프트 감지 시뮬레이션.

핵심 프로세스:
    1) 기준 통계 산출   - 학습에 쓴 판매 데이터의 "요일별 평균 판매수량" 계산
    2) 정상 입력 테스트 - 같은 분포의 데이터로 예측 -> RMSE 13개 이내 확인 (베이스라인)
    3) 드리프트 데이터 생성 - 변동성을 인위적으로 3배 키운 판매 데이터 생성
    4) 드리프트 데이터 주입 - 생성한 데이터를 서빙 서버에 연속 요청으로 전송
    5) 결과 관찰       - RMSE 상승 -> 알림 로그 발생 -> 재학습 트리거 확인

사전 준비: uvicorn serving_app.main:app 서버가 이미 떠 있어야 합니다.
실행: python scripts/simulate_drift.py
"""
import datetime
import os
import statistics
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import load_rows
from data.storage import latest_upload

API_URL = "http://localhost:8077/predict/batch-test"


def compute_weekday_profile(csv_path: str | None = None) -> list[float]:
    """1단계: 학습에 사용한 데이터(업로드된 최신 CSV)의 요일별 평균 판매수량.

    판매수량은 주가와 달리 "전날과 비슷"하지 않고 요일 주기가 지배적입니다
    (실측: 평일 61~65개 / 주말 41~44개). 그래서 평균 하나가 아니라 요일 7개의
    프로파일을 기준 통계로 삼습니다. 이벤트 기간(EventFlag=1)은 평상시 패턴이
    아니므로 제외합니다.
    """
    rows = load_rows(csv_path or latest_upload())
    by_weekday: dict[int, list[float]] = {}
    for r in rows:
        if r["EventFlag"] == 0:
            wd = datetime.date.fromisoformat(r["Date"]).weekday()
            by_weekday.setdefault(wd, []).append(r["SalesQty"])
    return [statistics.mean(by_weekday[d]) for d in range(7)]


# SEQ_LEN(20) + WINDOW_SIZE(21) = 41개를 보내야 배치 하나당 정확히 WINDOW_SIZE(21)개의
# (predicted, actual) 쌍이 쌓여, drift_detector.py가 바로 판정할 수 있다.
BATCH_N = 41

# 판매 데이터는 "요일 프로파일 x (1 + 노이즈)" 구조입니다. 실측 평상시 변동계수가
# 약 10%(평일 std 7 / 평균 64)라서 이를 정상 수준으로 두고, 드리프트는 원래 실습
# 시나리오와 똑같이 "변동성 3배"로 만듭니다.
NORMAL_SIGMA = 0.10
DRIFT_SIGMA = NORMAL_SIGMA * 3


def _seasonal_series(n: int, profile: list[float], sigma: float, start_weekday: int = 0) -> np.ndarray:
    noise = np.random.normal(0, sigma, n)
    base = np.array([profile[(start_weekday + i) % 7] for i in range(n)])
    return np.clip(base * (1 + noise), 1.0, None)  # 판매수량은 양수


def generate_normal_batch(profile: list[float], n=BATCH_N, sigma=NORMAL_SIGMA):
    """학습 데이터와 비슷한 변동성의 정상 입력 (요일 주기 + 10% 노이즈)."""
    return _seasonal_series(n, profile, sigma)


def generate_drift_batch(profile: list[float], n=BATCH_N, sigma=DRIFT_SIGMA):
    """변동성을 3배 키운 드리프트 입력 (의도적으로 오차 유발)."""
    return _seasonal_series(n, profile, sigma)


def send_batch(sales: np.ndarray, label: str) -> dict:
    resp = requests.post(API_URL, json={"sales": sales.tolist()})
    resp.raise_for_status()
    result = resp.json()
    print(f"[{label}] drift_check = {result['drift_check']}")
    return result


def main():
    profile = compute_weekday_profile()
    names = "월화수목금토일"
    print("[1] 기준 통계 (요일별 평균 판매수량): "
          + ", ".join(f"{names[d]}={profile[d]:.1f}" for d in range(7)))

    print("[2] 정상 입력 테스트 전송...")
    normal_batch = generate_normal_batch(profile)
    send_batch(normal_batch, label="normal")

    print("[3-4] 드리프트 입력 생성·주입...")
    drift_batch = generate_drift_batch(profile)
    send_batch(drift_batch, label="drift_injection")

    print("[5] 결과 확인: logs/aiops.log 또는 서버 콘솔에서 [WARN] drift detected 로그를 확인하세요.")


if __name__ == "__main__":
    main()
