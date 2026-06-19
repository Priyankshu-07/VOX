from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
    text: str = Field(..., example="traffic ki wajah se 10 min late honga")

class PredictResponse(BaseModel):
    intent: str
    confidence: float
    slots: Dict[str, str]
    response: str
