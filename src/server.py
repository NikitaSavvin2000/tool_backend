import os
import uvicorn
import pandas as pd
import multiprocessing
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from src.routers import router as api_router
from src.config import logger, public_or_local

load_dotenv()

logger.info("Starting microservice main forecast")

origins = ["http://localhost", "http://77.37.136.11"] if public_or_local == "LOCAL" else ["http://77.37.136.11"]

workers = multiprocessing.cpu_count()

print(f"[WORKERS] Count workers = {workers}")

security = HTTPBearer()
tokens_link = os.getenv("TOKEN_LIST")
tokens_df = pd.read_csv(tokens_link)
VALID_TOKENS = tokens_df[tokens_df["source"] == "tool_backend"]["token"].tolist()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized. To get access, contact @SavvinNikita on Telegram.")
    return token

docs_url= "/docs"
app = FastAPI(
    docs_url=docs_url,
    dependencies=[Depends(verify_token)]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Horizon System API"}


if __name__ == "__main__":
    port = 7071
    print(f'Documentation available at http://0.0.0.0:{port}{docs_url}')
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=workers, log_level="debug")
