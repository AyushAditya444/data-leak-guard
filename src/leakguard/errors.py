"""Errors in OpenAI's JSON shape, so OpenAI SDKs raise clean, readable exceptions."""

from fastapi.responses import JSONResponse


def openai_error(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )
