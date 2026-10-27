from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import os
from pathlib import Path

# Load pre‑computed forecasts once at startup
WAREHOUSE_PATH = Path(__file__).resolve().parents[2] / "warehouse" / "m5.duckdb"

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/forecast/{item_id}")
async def get_forecast(item_id: str):
    # Placeholder – replace with actual forecast loading logic
    if not WAREHOUSE_PATH.exists():
        raise HTTPException(status_code=500, detail="Warehouse not found")
    # Dummy response for demonstration
    return JSONResponse(content={"item_id": item_id, "forecast": []})

# Using FastAPI's lifespan context manager (Python 3.7+)
@app.on_event("startup")
async def startup_event():
    # You could load heavy models here if needed
    pass
