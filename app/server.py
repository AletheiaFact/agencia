from graph import WorkFlow
from fastapi import FastAPI, Request, Depends
from langserve import add_routes
from fastapi.responses import JSONResponse

# from middleware.auth_middleware import AuthMiddleware, verify_token
from fastapi.security import OAuth2PasswordBearer

agents_app = WorkFlow().app

app = FastAPI(
    title="Aletheia Server",
    version="1.0.0",
    description="aletheia API automatedFactChecking server using LangChain and CrewAI",
)


@app.post("/invoke")
async def stream(request: Request):
    req = await request.json()
    result = agents_app.invoke(req["input"])
    return JSONResponse(content={"message": result})


add_routes(app, agents_app)


# Health Check endpoint for k8s livenes probe
@app.get("/health", tags=["Health Check"])
async def health_check():
    # Here you might add logic to verify service health, like DB connection etc.
    return {"status": "ok"}


# app.add_middleware(AuthMiddleware)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

if __name__ == "__main__":
    import uvicorn

    # TODO host needs to be set through a env variable
    uvicorn.run(app, host="0.0.0.0", port=8080)
