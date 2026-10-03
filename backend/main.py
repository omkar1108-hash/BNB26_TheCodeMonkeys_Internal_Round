from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.upload import router as upload_router
from backend.api.analyze import router as analyze_router
from backend.api.bundle import router as bundle_router


app = FastAPI(
    title="TrustLayer",
    description="Multimodal digital content verification system",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(bundle_router)


@app.get("/")
def root():
    return {
        "project": "TrustLayer",
        "status": "running",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }