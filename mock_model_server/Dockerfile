FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mock

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

COPY requirements.txt .
RUN HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    NO_PROXY="${NO_PROXY}" \
    http_proxy="${http_proxy}" \
    https_proxy="${https_proxy}" \
    all_proxy="${all_proxy}" \
    no_proxy="${no_proxy}" \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
