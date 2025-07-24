# src/core/utils.py
import time
from functools import wraps
from fastapi import HTTPException, Request
from typing import Callable, Coroutine
from src.core.logger import logger
from starlette.requests import Request as StarletteRequest


def handle_exceptions(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as ve:
            logger.warning(f"Validation error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")
    return wrapper

def log_endpoint(logger=logger):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Поиск request в разных местах
            request = kwargs.get('request', None)
            
            if not request:
                for arg in args:
                    if isinstance(arg, (Request, StarletteRequest)):
                        request = arg
                        break
            
            if not request:
                for k, v in kwargs.items():
                    if isinstance(v, (Request, StarletteRequest)):
                        request = v
                        break

            try:
                if request:
                    logger.debug(
                        f"Request details: {request.method} {request.url}\n"
                        f"Headers: {dict(request.headers)}\n"
                        f"Path params: {request.path_params}\n"
                        f"Query params: {dict(request.query_params)}"
                    )
                    logger.info(
                        f"Request: {request.method} {request.url.path} "
                        f"from {request.client.host if request.client else 'unknown'}"
                    )
            except Exception as e:
                logger.error(f"Error logging request: {str(e)}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                if isinstance(result, Coroutine):
                    response = await result
                else:
                    response = result

                duration = time.time() - start_time

                logger.info(f"Response status: {getattr(response, 'status_code', 'unknown')} | Duration: {duration:.3f}s")
                logger.debug(f"Response details: {response}")

                return response
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__} | "
                    f"Duration: {duration:.3f}s\n"
                    f"Error details: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator