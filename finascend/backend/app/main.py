"""FastAPI application factory.

Run with:
    uvicorn app.main:app --reload
Interactive docs at /docs.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router

DESCRIPTION = """
Liquidity-management engine for small businesses.

The headline endpoint is **`GET /api/v1/simulation/runway-at-risk`**, which
reports Runway-at-Risk and CRaR: the liquidity analogues of VaR and CVaR,
computed from per-counterparty fitted payment-delay distributions coupled by a
Student-t copula.

Endpoints tagged *(not implemented)* return HTTP 501 with a stated reason.
They route, authenticate and validate correctly, but nothing behind them is
faked — no canned text, no placeholder numbers.

Get a token from `POST /api/v1/auth/token` and authorize with it.
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinAscend API",
        description=DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
    )

    # Both spellings of the loopback host are allowed by default. To a browser
    # `http://localhost:3000` and `http://127.0.0.1:3000` are DIFFERENT origins,
    # so allowing only one produces a frontend that works or fails depending on
    # which form you typed in the address bar — with the request blocked before
    # it ever reaches FastAPI, so the server log shows nothing at all.
    default_origins = ",".join(
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("FINASCEND_CORS_ORIGINS", default_origins).split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Uniform error envelope, per the architecture plan's Section 3.

        The exception text is included because this is a research/demo
        deployment where a debuggable failure is worth more than an opaque
        one. A production build would log the detail and return an opaque
        reference id instead.
        """
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "message": "An unexpected error occurred.",
                "details": {"exception": type(exc).__name__, "text": str(exc)[:500]},
            },
        )

    app.include_router(v1_router, prefix="/api/v1")

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
