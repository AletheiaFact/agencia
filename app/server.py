"""Agencia API server."""

from dotenv import load_dotenv

load_dotenv()

from graph import build_workflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Aletheia Server",
    version="2.0.0",
    description="Aletheia API automated fact-checking server using LangGraph",
)

workflow = build_workflow()


@app.post("/invoke")
async def invoke(request: Request):
    req = await request.json()
    result = workflow.invoke(req["input"])
    return JSONResponse(content={"message": result})


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
