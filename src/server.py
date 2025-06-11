# src/server.py
from fastapi import FastAPI
from src.api.v1.router import router as v1_router
from src.config import logger, public_or_local

logger.info("Starting microservice indicators")

app = FastAPI(docs_url="/backend/v1/", openapi_url='/backend/v1/openapi.json')

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost", "http://77.37.136.11"] if public_or_local == "LOCAL" else ["http://77.37.136.11"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем маршруты
app.include_router(v1_router, prefix="/backend/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}

if __name__ == "__main__":
    import uvicorn
    port = 7071
    print(f'Documentation available at http://0.0.0.0:{port}/backend/v1/')
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=False)