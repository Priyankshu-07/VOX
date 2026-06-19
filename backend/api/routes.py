from fastapi import APIRouter
from schemas.predict import PredictRequest, PredictResponse
from services.inference_service import inference_engine

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Lightweight health check endpoint to verify model load status.
    """
    return {
        "status": "online", 
        "model_loaded": inference_engine.ort_session is not None
    }

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Core NLU prediction endpoint.
    Accepts text, outputs intent classification, confidence score, and extracted slots.
    """
    result = inference_engine.predict(request.text)
    
    return PredictResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        slots=result["slots"],
        response=result["response"]
    )
