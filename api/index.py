"""
Vercel serverless function wrapper for FastAPI backend
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import the FastAPI app
from main import app
from mangum import Mangum

# Mangum adapter for AWS Lambda/Vercel
handler = Mangum(app, lifespan="off")

