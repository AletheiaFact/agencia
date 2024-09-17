FROM python:3.11-slim

WORKDIR /code

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["python", "/code/app/server.py"]
