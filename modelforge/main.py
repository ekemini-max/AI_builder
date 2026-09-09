import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from modelforge.db.database import init_db
from modelforge.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="ModelForge Platform",
    description="AutoML Training & Model Serving Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

os.makedirs("modelforge/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="modelforge/static"), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ModelForge"}
