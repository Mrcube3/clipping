import logging

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

logger = logging.getLogger("api.index")

try:
    from smart_reframe import app as smart_app
except Exception:
    smart_app = None
    logger.exception("Failed to import smart_reframe")

if smart_app is None:
    async def error_response(request):
        return PlainTextResponse(
            "Application failed to import. Check server logs for details.",
            status_code=500,
        )

    app = Starlette(debug=True, routes=[Route("/{path:path}", error_response)])
else:
    app = Starlette(debug=True, routes=[Mount("/", app=smart_app)])
