# Imagen única de inferencia (R7 batch y R8 online) para SageMaker.
# - `serve` (por defecto): servidor HTTP en :8080 con /ping e /invocations (Batch Transform y endpoint).
# - `prepare` / `publish`: pasos Processing del pipeline batch.
# El modelo NO va dentro de la imagen: se carga del Registry de MLflow por alias al arrancar.
# Base: misma imagen fijada por digest que usan los laboratorios del curso (S11/S12).
FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app
COPY requirements.txt requirements-dev.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY config ./config

# SageMaker ejecuta `docker run <imagen> serve` en Batch Transform y en endpoints.
ENTRYPOINT ["python", "-m", "bank_attrition.container"]
CMD ["serve"]
