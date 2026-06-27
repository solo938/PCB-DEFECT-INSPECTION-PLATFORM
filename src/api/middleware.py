# src/api/middleware.py
"""
Middleware for request logging, CORS, and rate limiting.
"""

import time
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all requests with latency, status, and client info.
    """
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        logger.info(f"REQUEST: {request.method} {request.url.path} from {client_ip}")
        
        try:
            response = await call_next(request)
            
            # Log response
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"RESPONSE: {request.method} {request.url.path} "
                f"-> {response.status_code} ({elapsed_ms:.1f}ms)"
            )
            
            return response
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"ERROR: {request.method} {request.url.path} "
                f"-> {str(e)} ({elapsed_ms:.1f}ms)"
            )
            raise


def add_cors_middleware(app):
    """Add CORS middleware to the app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app