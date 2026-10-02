"""Day1: 헬스체크 엔드포인트. (응답 형식은 기획서 5-2절 "헬스 체크")"""
from fastapi import APIRouter

from serving_app import model_loader

router = APIRouter()


@router.get("/health")
def health():
    model = model_loader._model_cache
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "loading_mode": _current_loading_mode(),
        "model_version": model.version if model is not None else None,  # lazy 모드에서 첫 /predict 전에는 null
    }


def _current_loading_mode() -> str:
    import os

    return os.getenv("LOADING_MODE", "lazy")
