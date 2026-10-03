from fastapi import FastAPI

from backend.api.upload import router as upload_router
from backend.api.analyze import router as analyze_router


app = FastAPI(
    title="TrustLayer",
    description="Multimodal digital content verification system",
    version="0.1.0"
)


app.include_router(upload_router)
app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "project": "TrustLayer",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }