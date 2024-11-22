FROM python:3.11-slim

WORKDIR /code

COPY . .

# Create the logs folder at the root to avoid app crashing
RUN mkdir -p /code/logs

RUN pip install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["python", "/code/app/server.py"]
