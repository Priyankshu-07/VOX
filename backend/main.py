from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from core.exceptions import global_exception_handler, inference_exception_handler, InferenceException
from core.logger import get_logger
import time
logger = get_logger("VOX_api")
app = FastAPI(
    title="VOX API",
    description="Offline Hinglish Delivery Assistant Backend",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    if "/health" not in request.url.path:
        logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f}ms with status {response.status_code}")  
    return response
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(InferenceException, inference_exception_handler)
app.include_router(router, prefix="/api")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
