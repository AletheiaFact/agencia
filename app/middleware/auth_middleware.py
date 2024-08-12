import os
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException, status
from jose import JWTError, jwt
from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

def verify_token(token: str):
    expired_token_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token has expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    invalid_token_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    malformed_token_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Malformed token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise invalid_token_exception
    except jwt.ExpiredSignatureError:
        raise expired_token_exception
    except Exception:
        print(f"Error verifying token")
        raise malformed_token_exception


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        authorization: str = request.headers.get("Authorization")

        if authorization:
            try:
                token_type, token = authorization.split()
                if token_type.lower() != "bearer":
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication scheme.",
                    )
                verify_token(token)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization header format.",
                )
            except JWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token.",
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication error.",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header is missing.",
            )
        
        response = await call_next(request)
        return response