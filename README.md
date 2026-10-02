#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : architect@sk.com, audit@korea.ac.kr 임성열 Ph.D.

# 신선식품 판매량 모델 서빙 및 AIOps 3일 실습 스켈레톤

신선식품 매장의 일별 판매 데이터로 다음날 판매수량을 예측하는
**LSTM** 모델을 Day1(서빙) → Day2(MLOps) → Day3(AIOps) 순서로 하나의 서빙 서버 위에
쌓아 올리는 실습 스켈레톤입니다. 데이터는 미리 생성해두지 않고, 대시보드에서 CSV
파일을 업로드하는 방식으로 공급합니다 - 실전에서 "새 데이터가 들어온다"는 상황을
그대로 흉내 낸 것입니다.

완성된 전체 기능(4탭 대시보드, 운영 지표, 알람 등)을 보고 싶다면 별도로 제공되는
**데모 패키지**(`project_answer_demo/`)를 참고하세요. 이 스켈레톤과 정답지는 실습
난이도를 낮추기 위해 핵심 루프(업로드 → 학습/서빙 → 드리프트 감지 → 재학습)만
남기고 나머지는 들어내 뒀습니다.

## 데이터 - 대시보드에서 업로드

`data/sample_fresh_sales.csv`는 **2023-01-01 ~ 2025-02-06**의 일별 판매 기록
(768일 ≈ 2년, `Date,SalesQty,OrderQty,WasteQty,EventFlag` 컬럼)입니다. 모델이
쓰는 것은 **`SalesQty`와 `EventFlag` 2개뿐**이고, `OrderQty`/`WasteQty`는
`data/features.py`의 `load_rows()`에서 읽지 않고 버립니다. `Date`도 모델 입력이
아니라 "행이 날짜 오름차순"이라는 전제와 대시보드 표시에만 씁니다.

데이터의 지배적인 구조는 **요일 주기**입니다 (평상시 평일 61~65개 / 주말 41~44개,
변동계수 약 6%). 여기에 전체의 약 22%를 차지하는 이벤트 기간(`EventFlag=1`,
평균 81.6개)이 겹칩니다. 주가와 달리 "전날과 비슷"하지 않다는 점이 중요합니다 -
전날값을 그대로 예측하면 RMSE 15.1로, 전체 평균을 찍는 것(15.6)과 차이가 없습니다.

`EventFlag`는 **이벤트 기간과 그 직전 1일(사전 공지일)**에 1입니다. 프로모션은
현실에서도 사전에 계획되므로 하루 전에는 알 수 있다는 가정입니다. 이 1비트가
중요한 이유는 모델의 구조적 한계와 맞닿아 있습니다 - `build_sequences()`는 t+1일을
예측할 때 t일까지의 `EventFlag`만 보기 때문에, 플래그가 이벤트 당일에만 서 있으면
직전 20일이 전부 `EventFlag=0`인 상태에서 판매량이 뛰는 **첫날을 원리적으로 맞힐
수 없습니다**. 사전 공지일 덕분에 모델은 "내일 이벤트가 시작된다"를 입력으로 받습니다.

서버를 띄운 뒤 대시보드(`http://localhost:8077/`)의 업로드 카드에서 이 파일을
그대로 올리면 됩니다. 업로드된 CSV는 `data/uploads/`에 타임스탬프 파일명으로
쌓이고, 학습·시뮬레이션 코드는 항상 **가장 최근에 업로드된 파일**을 사용합니다
(`data/storage.py`의 `latest_upload()`). 여러 번 업로드하면 그때마다 최신 파일로
전환되므로, 다른 판매 CSV(최소 `Date,SalesQty,EventFlag` 컬럼)로 바꿔 실험해볼
수도 있습니다.

## 모델 아키텍처

최근 20일(SEQ_LEN)의 (판매수량, 이벤트플래그) 시퀀스를 입력받아 다음날 판매수량을
예측하는 3층 LSTM입니다.

```
Input (20, 2)  ->  LSTM(32, return_sequences=True)  ->  LSTM(32, return_sequences=True)
               ->  LSTM(16)  ->  Dense(16, relu)  ->  Dense(1)
```

2년치(~768일) 데이터 + SEQ_LEN(20)을 적용하면 학습 시퀀스가 약 598개까지 늘어나,
파라미터(약 1.6만 개) 대비 샘플 비율이 충분히 확보됩니다. 그래서 층을 깊게(LSTM 3층)
쌓았습니다 - CPU로 100 epoch을 학습해도 1분 내외면 끝납니다. (아키텍처 정의는
`serving_app/lstm_model.py`, Day1·Day2가 공유합니다.)

재학습 방식도 유의해서 보세요. Day3에서 드리프트가 감지되면 **처음부터 다시 학습하지
않습니다.** 최근 3주(21일)만으로 LSTM을 스크래치로 학습시키기엔 샘플이 너무
적어 불안정하기 때문에, 이미 전체 데이터로 학습된 **Production 가중치에서 이어서
(warm start) 짧게(10 epoch) fine-tuning**합니다. `serving_app/train_and_register.py`의
`train_and_register()`(Day2, 처음부터 학습)와 `fine_tune()`(Day3, 이어서 학습)이 이 구분입니다.

### RMSE 게이트가 10.00개인 이유

게이트 값은 "단순 베이스라인"과 "달성 가능한 상한" 사이에서 잡습니다. LSTM과
**같은 입력(20일 x 2채널)**을 준 모델들을 holdout(뒤 20%, 150건)으로 실측한 결과입니다.

| 예측기 | RMSE | 성격 |
|---|---|---|
| 전체 평균 (상수) | 15.59 | 베이스라인 |
| 전날값 그대로 (persistence) | 15.11 | 베이스라인 |
| 요일 평균 | 13.59 | 베이스라인 |
| **← 게이트 10.00** | | |
| 선형회귀 (LSTM과 동일 입력) | 7.74 | 상한 |
| GBM (LSTM과 동일 입력) | 7.07 | 상한 |
| 요일 + *당일* EventFlag (미래 정보 치트) | 6.75 | 상한 |

**10.00**은 요일 평균(13.59)보다 확실히 낫고 상한(7.07~7.74)에는 아직 못 미치는
위치입니다. 즉 "단순 규칙만으로 되는 모델은 막고, 상한에 근접한 모델만 통과"시킵니다.
base 학습의 실측 RMSE는 약 9로 통과합니다.

`train_and_register.py`의 `RMSE_GATE`와 `monitoring/drift_detector.py`의
`RMSE_THRESHOLD`는 **반드시 같은 값**이어야 합니다 - 다르면 "드리프트로 재학습을
걸었는데 게이트는 통과 못 해 영원히 승격 안 되는" 구간이 생깁니다.

> **참고 - 데이터셋이 게이트 10에 맞춰 보정되어 있습니다.**
> 원본 데이터에서는 상한이 선형회귀 10.95 / GBM 11.19였고 base LSTM은 약 12여서,
> 10.00은 어떤 모델도 넘지 못하는 선이었습니다. 원인은 두 가지였습니다 -
> ① `EventFlag`가 이벤트 당일만 표시해 시작일을 맞힐 수 없었고(오차 -20~-33이 5건),
> ② 평상시 변동계수가 약 11%로 노이즈 하한 자체가 높았습니다.
> 그래서 `data/sample_fresh_sales.csv`를 **요일/이벤트 구조는 그대로 두고**
> ① 사전 공지일 1일 추가, ② 잔차를 0.6배로 축소(CV 11% → 6%)하여 보정했습니다.
> 날짜 범위, 요일 프로파일, 이벤트 달력·길이·배율(1.496), `OrderQty`는 원본과 같습니다.

## 디렉토리 구조

```
project2/
├── requirements.txt
├── data/
│   ├── sample_fresh_sales.csv  # 대시보드에 업로드해볼 예시 데이터 (2년치 일별 판매)
│   ├── storage.py               # 업로드된 CSV 중 최신 파일을 찾는 latest_upload()
│   ├── uploads/                 # 업로드된 CSV가 쌓이는 곳 (시작 시 비어 있음)
│   └── features.py              # 시퀀스 빌더(SEQ_LEN=20) + SalesScaler (전 Day 공용)
├── scripts/                    # 서빙 앱 밖에서 실행하는 실습/시뮬레이션 도구
│   ├── train_baseline_v1.py    # Day1 사전 준비: MLflow 없이 로컬 baseline LSTM 생성
│   └── simulate_drift.py       # Day3: 정상/드리프트 배치 생성 + 서버로 주입
└── serving_app/
    ├── main.py                     # Day1 - app 생성, 라우터 등록, 로딩 모드 분기
    ├── schemas.py                  # Day1
    ├── lstm_model.py                # Day1·Day2 공유 아키텍처 정의
    ├── model_loader.py             # Day1 → Day2(MLflow 연동)
    ├── train_and_register.py       # Day2 (base 학습) + Day3 (fine-tuning)
    ├── Dockerfile, docker-compose.yml   # Day2 (단일 컨테이너)
    ├── models/
    │   ├── fresh_sales_v1.keras    # Day1 로컬 baseline 모델 (train_baseline_v1.py가 생성)
    │   └── scaler.pkl              # Day1~3 공용 정규화 스케일러 (train_baseline_v1.py가 생성)
    ├── routers/
    │   ├── predict.py              # Day1 → Day3(시뮬레이션 엔드포인트 추가)
    │   ├── health.py               # Day1
    │   ├── data.py                 # Day2: CSV 업로드 (완성형)
    │   └── logs.py                 # Day3: logs/aiops.log 파일 조회 (완성형)
    ├── monitoring/                 # Day3
    │   ├── drift_detector.py       # TODO
    │   └── retrain_trigger.py      # TODO
    ├── static/
    │   └── index.html              # 실습용 대시보드 (완성형) - http://localhost:8000/
    └── logs/                       # retrain_trigger.py의 "aiops" 로거가 쓰는 곳 (실행 시 자동 생성)
```

`serving_app/routers/data.py`, `routers/logs.py`, `static/index.html`은 실습
목표가 아니라 업로드·결과 확인을 위한 배관 코드라 처음부터 완성된 형태로
제공됩니다 - 전부 방금 업로드된 파일이나 로그 파일처럼 이미 존재하는 데이터를
읽거나 저장할 뿐, 가짜 데이터를 만들지 않습니다. 재학습 이력을 MLflow Model
Registry API로 따로 조회하는 대신, `retrain_trigger.py`가 남기는 로그 파일을
그대로 보여주는 쪽을 택했습니다 - 학생이 봐야 할 것은 "재학습이 실제로
일어났다는 증거"이지 레지스트리 조회 API 설계가 아니기 때문입니다.

## 실습용 대시보드

`http://localhost:8077/`은 개발자 대시보드입니다.

- **판매 데이터 업로드** - `data/sample_fresh_sales.csv`(또는 같은 형식의 다른 CSV)를
  올리면 `/data/upload`로 전송되고, 업로드 완료 여부가 그 자리에 바로 표시됩니다.
- **드리프트 시뮬레이션** - 정상/드리프트 배치를 `/predict/batch-test`로 전송합니다.
  `batch_test()`가 TODO인 동안은 응답이 비어 있거나 오류가 납니다.
- **드리프트 감지 기반 재학습 파이프라인** - 감지 → fine-tuning → 재배포 단계를
  시각화합니다. `drift_detector.py`·`retrain_trigger.py`의 TODO를 채우기 전까지는
  단계가 진행되지 않습니다.
- **재학습 로그** - `/logs`로 `logs/aiops.log` 파일 목록을 보여주고, 클릭하면
  `/logs/{파일명}`으로 내용을 그대로 열어 보여줍니다. `[WARN] drift detected` →
  `[INFO] retrain triggered` → `[OK] new_rmse=...`가 순서대로 쌓이는지 직접
  확인하는 용도입니다.

## 실습 시나리오 (Day1 → Day2 → Day3)

**Day1 — 로컬 baseline LSTM을 FastAPI로 서빙**
Lazy/Eager 로딩 비교, `/predict`·`/health` 동작 확인, `http://localhost:8077/`에서 대시보드로 확인

**Day2 — Day1 서버 + MLflow 학습·레지스트리·컨테이너화**
base 학습(scratch, 100 epoch), RMSE 게이트(10.00개) 통과 버전만 Production 승격, Docker로 재현

**Day3 — Day2 서버에 드리프트 감지·자동 재학습 부착**
드리프트 주입 → fine-tuning(warm start, 10 epoch) → 자동 재배포 확인

세 Day의 산출물은 독립적이지 않고 **하나의 `serving_app/`** 위에 순서대로 쌓입니다
(`model_loader.py`가 Day1 로컬 모델 → Day2 MLflow Production 모델로 전환되는 지점이 그 연결고리입니다).
스케일러(`scaler.pkl`)는 Day1에서 한 번 fit한 뒤 Day1~3 내내 그대로 재사용됩니다 -
fine-tuning 시 다시 fit하면 이미 그 스케일로 학습된 기존 가중치와 어긋나기 때문입니다.

## 실행 순서

```bash
pip install -r requirements.txt

# --- Day1 ---
uvicorn serving_app.main:app --host 0.0.0.0 --port 8077 # http://localhost:8077/ 대시보드, /docs 에서 API 확인
# (lazy 모드가 기본이라 업로드된 데이터가 없어도 서버는 정상적으로 뜹니다)

# 대시보드 업로드 카드에서 data/sample_fresh_sales.csv 를 업로드한 뒤,
# 별도 터미널에서:
python scripts/train_baseline_v1.py       # 로컬 baseline LSTM + scaler.pkl 생성
# LOADING_MODE=eager uvicorn serving_app.main:app --reload  # Eager 방식과 시작 시간 비교

# --- Day2 ---
python serving_app/train_and_register.py            # 로컬 MLflow(sqlite)에 학습 기록 + 게이트 통과 시 Production 승격
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077 

# --- 컨테이너로 재현 (단일 컨테이너 - MLflow도 서버 없이 컨테이너 안에서 로컬로 동작) ---
docker compose -f serving_app/docker-compose.yml up --build
# (샘플 데이터를 컨테이너 안에 "업로드"해 둔 뒤 베이스라인 -> MLflow 학습/등록까지
#  이미지 빌드 시점에 전부 끝나므로, 로컬에서 미리 실행해둘 필요가 없습니다)

# --- Day3 ---
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
# (MODEL_SOURCE=mlflow 로 띄워야 재학습으로 승격된 새 버전이 /predict 에 바로 반영됩니다 - local 이면 계속 v1-local)
python scripts/simulate_drift.py          # 정상 배치 → 드리프트 배치 순서로 주입
```

`/predict`는 단일 값이 아니라 **최근 20일치 시퀀스**와 **현재 재고**를 받아, 다음날 예측 판매량과
권장 발주량을 돌려줍니다 (기획서 5-2절). `/docs`의 "Try it out"에는 샘플 데이터 마지막 20일이 예시로 채워져 있습니다.

```json
{
  "sequence": [
    {"sales_qty": 62.0, "event_flag": 0},
    {"sales_qty": 64.0, "event_flag": 0},
    { "...": "18개 더" }
  ],
  "current_stock": 9,
  "prediction_date": "2026-01-02"
}
```

```json
{
  "prediction_date": "2026-01-02",
  "predicted_sales": 48,
  "safety_stock": 5,
  "current_stock": 9,
  "recommended_order": 44,
  "model_version": "production-v3",
  "status": "success"
}
```

- 권장 발주량 = 예측 판매량 + 안전재고(예측의 10%, 올림) - 현재 재고 (0 미만이면 0). `prediction_date`를 생략하면 내일 날짜.
- 입력이 잘못되면 422 (메시지는 한국어, `serving_app/errors.py`), 모델을 불러오지 못하면 503.
- `/predict/batch-test` 응답에는 기존 `predictions`·`drift_check`에 더해 `rmse`·`threshold`·`status`·`message`·`model_version`이 들어갑니다.
- 모든 응답에 처리 시간 헤더 `X-Process-Time`이 붙고, `/predict`가 1초를 넘으면 `logs/aiops.log`에 `[WARN] slow response`가 남습니다.

## TODO 체크리스트 (실습생이 채워야 하는 부분)

이 스켈레톤은 배관(라우팅·업로드·MLflow 학습 로직 등)은 완성되어 있고,
**각 Day의 핵심 학습 목표에 해당하는 부분만 TODO로 비워뒀습니다.**

- `serving_app/model_loader.py` → `_load_from_mlflow()`: MLflow Production 모델 로드 (Day2)
- `serving_app/routers/predict.py` → `batch_test()`: 슬라이딩 윈도우 예측 + `recent_predictions` 누적 (Day3)
- `serving_app/monitoring/drift_detector.py` → `compute_rmse()`: RMSE 직접 구현 (Day3)
- `serving_app/monitoring/retrain_trigger.py` → `check_and_trigger()`: fine-tuning 트리거 연결 (Day3, 힌트: 파일 상단 주석 참고)
- `scripts/simulate_drift.py` → `send_batch()`: `/predict/batch-test` 호출 (Day3)

## 완료 기준

- [ ] `/data/upload`로 CSV를 올리면 업로드 완료로 표시되는가 (`/data/status`로도 확인 가능)
- [ ] 정상 데이터로는 RMSE 10개 이내, 드리프트 데이터로는 10개 초과가 재현되는가
- [ ] `logs/aiops.log`에 `[WARN] drift detected` → `[INFO] retrain triggered` →
      `[OK] new_rmse=...` 순서로 기록되는가
- [ ] 재배포 후 `/predict` 호출 시 새 Production 버전이 응답하는가
