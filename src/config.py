# src/config.py
import os
from src.core.logger import setup_logger

logger = setup_logger()
public_or_local = os.getenv("PUBLIC_OR_LOCAL", "LOCAL")