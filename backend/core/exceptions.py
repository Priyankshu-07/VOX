from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("edge_assist_api")

class InferenceException(Exception):
    def __init__(self, message: str):
        self.message = message

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)}
    )

async def inference_exception_handler(request: Request, exc: InferenceException):
    logger.error(f"Inference error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Inference Error", "message": exc.message}
    )
