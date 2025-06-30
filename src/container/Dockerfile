FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt /app/

RUN python -m venv .venv

RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Create an anonymous volume to override the bind mount on .venv file
VOLUME ["/app/.venv"]

CMD ["/app/.venv/bin/python", "./container/main.py"]
