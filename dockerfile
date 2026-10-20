FROM python:3.12-slim

# uv, para instalar dependencias igual que en local
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Primero solo las dependencias: Docker cachea esta capa
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Después el código y el modelo
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml README.md ./

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "emociones.api:app", "--host", "0.0.0.0", "--port", "8000"]