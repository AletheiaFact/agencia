from graph import WorkFlow
from fastapi import FastAPI
from langserve import add_routes

agents_app = WorkFlow().app

app = FastAPI(
    title="Aletheia Server",
    version="1.0.0",
    description="aletheia API automatedFactChecking server using LangChain and CrewAI",
)

add_routes(
    app,
    agents_app
)

# Health Check endpoint for k8s livenes probe
@app.get("/health", tags=["Health Check"])
async def health_check():
    # Here you might add logic to verify service health, like DB connection etc.
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # TODO host needs to be set through a env variable
    uvicorn.run(app, host="0.0.0.0", port=8080)
