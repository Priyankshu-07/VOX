from fastapi import APIRouter
from schemas.predict import PredictRequest, PredictResponse
from services.inference_service import inference_engine
router = APIRouter()
@router.get("/health")
async def health_check():
    return {
        "status": "online", 
        "model_loaded": inference_engine.ort_session is not None
    }
@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    result = inference_engine.predict(request.text)
    return PredictResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        slots=result["slots"],
        response=result["response"]
    )
