"""
신선식품 판매 데이터 업로드 - 대시보드에서 CSV를 직접 올리는 방식입니다.

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.

오류 응답 (기획서 5-3절)
  400 : 파일 형식이 다름 (.csv 가 아니거나 UTF-8 이 아님)
  422 : 파일은 CSV 지만 내용이 학습에 쓸 수 없음 (필수 컬럼 누락, 행 수 부족, 판매량 음수 등)
대시보드는 detail 을 문자열 그대로 보여주므로, detail 은 한 줄짜리 한국어 메시지로 돌려줍니다.
"""
import csv
import io
import math
import os
import time
from datetime import date

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Date", "SalesQty", "EventFlag"}
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "CSV 파일(.csv)만 업로드할 수 있습니다.")
    raw = await file.read()
    try:
        # utf-8-sig: 엑셀에서 저장한 CSV 앞에 붙는 BOM 을 떼어냅니다 (안 떼면 첫 컬럼이 "\ufeffDate" 가 됨)
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise HTTPException(422, f"CSV에 필수 컬럼 {sorted(missing)}이(가) 없습니다. (필수: {sorted(REQUIRED_COLUMNS)})")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(422, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다. (현재 {len(rows)}행)")
    error = _find_invalid_row(rows)
    if error:
        raise HTTPException(422, error)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"sales_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


def _find_invalid_row(rows: list[dict]) -> str | None:
    """
    학습 코드(data/features.py)가 전제하는 조건을 업로드 시점에 미리 확인합니다. (기획서 3-1절 데이터 유효성 검증)
    첫 번째로 잘못된 행의 설명을 돌려주고, 문제가 없으면 None. 줄 번호는 헤더를 1번째 줄로 셉니다.
      · Date      : YYYY-MM-DD 형식, 오래된 날짜부터 오름차순·중복 없음 (시퀀스가 시간 순서라는 전제)
      · SalesQty  : 0 이상의 숫자
      · EventFlag : 0 또는 1
    """
    prev_date = None
    for i, r in enumerate(rows):
        line = i + 2
        raw_date, raw_sales, raw_flag = (str(r.get(c) or "").strip() for c in ("Date", "SalesQty", "EventFlag"))

        try:
            d = date.fromisoformat(raw_date)
        except ValueError:
            return f"{line}번째 줄: Date는 YYYY-MM-DD 형식이어야 합니다. (값: '{raw_date}')"
        if prev_date is not None and d <= prev_date:
            return f"{line}번째 줄: Date는 오래된 날짜부터 오름차순이어야 하고 중복이 없어야 합니다. ({prev_date} 다음에 {d})"
        prev_date = d

        try:
            sales = float(raw_sales)
        except ValueError:
            sales = math.nan
        if not math.isfinite(sales) or sales < 0:
            return f"{line}번째 줄: SalesQty(판매량)는 0 이상의 숫자여야 합니다. (값: '{raw_sales}')"

        try:
            flag = float(raw_flag)
        except ValueError:
            flag = math.nan
        if flag not in (0.0, 1.0):
            return f"{line}번째 줄: EventFlag(이벤트 여부)는 0 또는 1이어야 합니다. (값: '{raw_flag}')"
    return None


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    sales = [r["SalesQty"] for r in rows]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_sales_qty": min(sales),
        "max_sales_qty": max(sales),
        "event_days": int(sum(r["EventFlag"] for r in rows)),
    }
