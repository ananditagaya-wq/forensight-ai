import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import initialize_database
from app.routes import CSRF_COOKIE, router


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="ForenSight AI",
    description="Forensic evidence handling risk assessment and recordkeeping.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


def _set_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")


@app.middleware("http")
async def protect_unsafe_requests(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        cookie_token = request.cookies.get(CSRF_COOKIE, "")
        header_token = request.headers.get("X-CSRF-Token", "")
        if not cookie_token or not header_token or not hmac.compare_digest(
            cookie_token, header_token
        ):
            response = JSONResponse(
                status_code=403,
                content={"detail": "Missing or invalid CSRF token."},
            )
            _set_security_headers(response)
            return response
        origin = request.headers.get("origin")
        if origin:
            expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if not hmac.compare_digest(origin.rstrip("/"), expected_origin.rstrip("/")):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request rejected."},
                )
                _set_security_headers(response)
                return response
    response = await call_next(request)
    _set_security_headers(response)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
