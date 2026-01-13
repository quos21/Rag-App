import os
from typing import List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

class XAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        protected_paths: Optional[List[str]] = None,
        env_var_name: str = "X_AUTH_TOKEN",
        allow_options: bool = True,
    ):
        super().__init__(app)
        self.protected_paths = protected_paths or []
        self.env_var_name = env_var_name
        self.allow_options = allow_options

    async def dispatch(self, request: Request, call_next):
        # Allow preflight OPTIONS requests
        if self.allow_options and request.method.upper() == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        
        # Check if path needs authentication
        if self._is_protected(path):
            expected_token = os.getenv(self.env_var_name)
            
            if not expected_token:
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Server missing {self.env_var_name}"},
                )
            
            # Get token from headers
            token = request.headers.get("x_auth_token")
            if not token:
                auth = request.headers.get("authorization")
                if auth and auth.lower().startswith("bearer "):
                    token = auth.split(" ", 1)[1].strip()
            
            if not token or token != expected_token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-Auth-Token"}
                )
        
        return await call_next(request)

    def _is_protected(self, path: str) -> bool:
        """Check if path requires authentication"""
        for protected in self.protected_paths:
            if path.startswith(protected):
                return True
        return False