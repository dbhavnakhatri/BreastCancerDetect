"""
Minimal Vercel serverless function - returns mock data
Use this temporarily while setting up Railway for the full backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

app = FastAPI(title="Breast Cancer Detection API - Minimal Version")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": False,
        "message": "This is a minimal version. Deploy full backend to Railway for ML functionality."
    }

@app.post("/api/analyze")
async def analyze():
    return JSONResponse(
        status_code=503,
        content={
            "detail": "ML model not available on Vercel. Please deploy backend to Railway. See VERCEL_DEPLOYMENT_NOTES.md"
        }
    )

@app.post("/api/report")
async def report():
    return JSONResponse(
        status_code=503,
        content={
            "detail": "ML model not available on Vercel. Please deploy backend to Railway. See VERCEL_DEPLOYMENT_NOTES.md"
        }
    )

handler = Mangum(app, lifespan="off")

