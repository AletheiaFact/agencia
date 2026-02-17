"""Agencia API server."""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

# Configure structured logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("agencia")

from graph import build_workflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Aletheia Server",
    version="2.0.0",
    description="Aletheia API automated fact-checking server using LangGraph",
)

workflow = build_workflow()
logger.info("Workflow compiled successfully. Nodes: %s", list(workflow.get_graph().nodes))


@app.post("/invoke")
async def invoke(request: Request):
    req = await request.json()
    input_data = req["input"]
    claim = input_data.get("claim", "")[:80]
    search_type = input_data.get("search_type", "online")
    language = input_data.get("language", "pt")
    logger.info("POST /invoke claim='%s' search_type=%s language=%s", claim, search_type, language)

    try:
        result = workflow.invoke(input_data)
        logger.info("POST /invoke completed successfully for claim='%s'", claim)
        return JSONResponse(content={"message": result})
    except Exception as e:
        logger.error("POST /invoke failed for claim='%s': %s", claim, e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
