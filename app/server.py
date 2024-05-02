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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)