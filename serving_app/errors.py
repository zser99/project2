"""
422(입력 검증 실패) 응답 메시지를 한국어로 바꾸는 예외 처리기.

FastAPI 는 schemas.py 의 검증에 실패하면 {"detail": [{"loc": ..., "msg": ..., "type": ...}]} 를 돌려주는데,
msg 가 "List should have at least 20 items after validation, not 3" 같은 영어 문장이라 점포 운영자나
프론트엔드가 그대로 보여주기 어렵습니다. 응답 구조(loc·type)는 그대로 두고 msg 만 한국어로 바꿉니다.
(기획서 5-3절 "오류 응답 예시")

main.py 에서 register_error_handlers(app) 로 등록합니다.
"""
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from data.features import SEQ_LEN

_SEQUENCE_LENGTH_MSG = f"최근 {SEQ_LEN}일의 판매 데이터가 필요합니다."
_SALES_NEGATIVE_MSG = "판매량은 0 이상이어야 합니다."
_EVENT_FLAG_MSG = "이벤트 여부(event_flag)는 0 또는 1이어야 합니다."

# (필드 이름, pydantic 오류 type) → 메시지.  필드 이름은 loc 의 마지막 문자열 (예: ["body", "sequence", 5, "sales_qty"] → "sales_qty")
_FIELD_MESSAGES = {
    ("sequence", "too_short"): _SEQUENCE_LENGTH_MSG,
    ("sequence", "too_long"): _SEQUENCE_LENGTH_MSG,
    ("sales_qty", "greater_than_equal"): _SALES_NEGATIVE_MSG,
    ("event_flag", "greater_than_equal"): _EVENT_FLAG_MSG,
    ("event_flag", "less_than_equal"): _EVENT_FLAG_MSG,
    ("current_stock", "missing"): "권장 발주량 계산을 위해 현재 재고가 필요합니다.",
    ("current_stock", "greater_than_equal"): "현재 재고는 0 이상이어야 합니다.",
    ("sales", "too_short"): f"드리프트 배치 테스트에는 판매량이 최소 {SEQ_LEN + 1}개 필요합니다.",
    ("sales", "greater_than_equal"): _SALES_NEGATIVE_MSG,
}

# 필드와 상관없이 오류 type 만으로 정하는 메시지
_TYPE_MESSAGES = {
    "float_parsing": "숫자여야 합니다.",
    "float_type": "숫자여야 합니다.",
    "int_parsing": "정수여야 합니다.",
    "int_type": "정수여야 합니다.",
    "int_from_float": "정수여야 합니다.",
    "date_parsing": "날짜는 YYYY-MM-DD 형식이어야 합니다.",
    "date_from_datetime_parsing": "날짜는 YYYY-MM-DD 형식이어야 합니다.",
    "date_type": "날짜는 YYYY-MM-DD 형식이어야 합니다.",
    "list_type": "목록(배열) 형식이어야 합니다.",
    "model_attributes_type": "객체 형식이어야 합니다.",
    "dict_type": "객체 형식이어야 합니다.",
    "json_invalid": "요청 본문이 올바른 JSON 형식이 아닙니다.",
}


def _korean_message(err: dict) -> str:
    field = next((x for x in reversed(err.get("loc", ())) if isinstance(x, str)), "")
    err_type = err.get("type", "")
    msg = _FIELD_MESSAGES.get((field, err_type)) or _TYPE_MESSAGES.get(err_type)
    if msg is None and err_type == "missing":
        msg = f"필수 항목 '{field}' 값이 없습니다."
    if msg is None:
        return err.get("msg", "")  # 따로 정하지 않은 오류는 원래 메시지 그대로

    ctx = err.get("ctx") or {}
    if err_type in ("too_short", "too_long") and "actual_length" in ctx:
        msg += f" (받은 개수: {ctx['actual_length']})"
    return msg


async def _validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # RequestValidationError 에만 등록하므로 항상 참
    detail = [
        {"loc": err.get("loc"), "msg": _korean_message(err), "type": err.get("type"), "input": err.get("input")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": detail}))


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
