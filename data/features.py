"""
신선식품 판매 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

Day1 baseline 학습(scripts/train_baseline_v1.py), Day2 MLflow 학습
(serving_app/train_and_register.py), Day3 fine-tuning 재학습
(monitoring/retrain_trigger.py)이 모두 이 모듈을 재사용합니다. 시퀀스 정의를
한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는 실무 사고를
방지할 수 있습니다.

입력 시퀀스: 최근 SEQ_LEN(20)일의 (sales_qty, event_flag)
타깃: 그다음 날의 sales_qty

CSV의 OrderQty/WasteQty는 읽지 않고 버립니다(모델이 쓰지 않는 컬럼).
Date도 모델 입력이 아니며, 행이 날짜 오름차순이라는 전제와 대시보드 표시에만 씁니다.
"""
import csv
import pickle

SEQ_LEN = 20  # LSTM 입력 윈도우 길이 (일수) - 주간 주기(7일)를 약 3회 포함
_NOT_FITTED_MSG = "SalesScaler가 아직 fit/load되지 않았습니다 (scripts/train_baseline_v1.py로 scaler.pkl을 먼저 만드세요)."


def load_rows(csv_path: str = "data/sample_fresh_sales.csv") -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "Date": r["Date"],
                "SalesQty": float(r["SalesQty"]),
                "EventFlag": float(r["EventFlag"]),
            }
            for r in reader
        ]
    return rows


class SalesScaler:
    """
    sales_qty/event_flag를 각각 [0, 1] 범위로 정규화하는 min-max 스케일러.

    LSTM은 스케일에 민감하기 때문에(트리 기반 모델과 달리) 반드시 정규화가 필요합니다.
    Day1에서 base 데이터로 한 번 fit한 뒤 serving_app/models/scaler.pkl로 저장해두고,
    Day2 MLflow 학습과 Day3 fine-tuning 모두 같은 스케일러를 재사용합니다.
    (fine-tuning 시 스케일러를 다시 fit하지 않는 이유: 이미 이 스케일로 학습된 모델
     가중치와 어긋나면 fine-tuning 자체가 무의미해지기 때문입니다.)

    event_flag는 원래 0/1이라 min=0, max=1 -> 정규화해도 값이 그대로입니다.
    """

    def __init__(self):
        self.sales_min = self.sales_max = None
        self.event_min = self.event_max = None

    def fit(self, rows: list[dict]) -> "SalesScaler":
        sales = [r["SalesQty"] for r in rows]
        events = [r["EventFlag"] for r in rows]
        self.sales_min, self.sales_max = min(sales), max(sales)
        self.event_min, self.event_max = min(events), max(events)
        return self

    def _scale(self, value: float, lo: float | None, hi: float | None) -> float:
        if lo is None or hi is None:
            raise RuntimeError(_NOT_FITTED_MSG)
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _unscale(self, value: float, lo: float | None, hi: float | None) -> float:
        if lo is None or hi is None:
            raise RuntimeError(_NOT_FITTED_MSG)
        return value * (hi - lo) + lo

    def transform_point(self, sales_qty: float, event_flag: float) -> list[float]:
        return [
            self._scale(sales_qty, self.sales_min, self.sales_max),
            self._scale(event_flag, self.event_min, self.event_max),
        ]

    def scale_sales(self, sales_qty: float) -> float:
        """타깃(다음날 판매수량)을 학습용으로 정규화. 입력 시퀀스와 같은 스케일을 써야
        손실(loss)이 과도하게 커지지 않고 학습이 안정적으로 수렴한다."""
        return self._scale(sales_qty, self.sales_min, self.sales_max)

    def inverse_sales(self, scaled_sales: float) -> float:
        """모델이 뱉은 정규화된 예측값을 실제 개수 단위 판매수량으로 되돌린다."""
        return self._unscale(scaled_sales, self.sales_min, self.sales_max)

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "SalesScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def build_sequences(rows: list[dict], scaler: SalesScaler, seq_len: int = SEQ_LEN):
    """
    rows(시간순 일별 판매 데이터)에서 (SEQ_LEN, 2) 크기의 정규화된 입력 시퀀스와
    다음날 판매수량(정규화 전 실값) 타깃을 만든다.

    반환: X (n_samples, seq_len, 2), y (n_samples,) - y는 스케일 안 된 실제 판매수량
    """
    scaled_points = [scaler.transform_point(r["SalesQty"], r["EventFlag"]) for r in rows]
    sales = [r["SalesQty"] for r in rows]

    X, y = [], []
    for i in range(len(rows) - seq_len):
        X.append(scaled_points[i : i + seq_len])
        y.append(sales[i + seq_len])
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2):
    """시간 순서를 유지한 채 앞부분을 train, 뒷부분을 test로 나눈다 (미래 데이터 누수 방지)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
