FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app  

# xgboost runtime (if you use xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the whole project (keeps app/ and model/ structure)
COPY . .

# Optional: test stage adds dev deps and runs tests
FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
ENV XGB_MODEL_PATH=/app/model/titanic_xgboost_pipeline.joblib \
    PKL_MODEL_PATH=/app/model/titanic_pipeline.pkl
CMD ["pytest", "-q"]

# Final image used in prod
FROM base AS runtime
ENV FLASK_RUN_HOST=0.0.0.0 \
    XGB_MODEL_PATH=/app/model/titanic_xgboost_pipeline.joblib \
    PKL_MODEL_PATH=/app/model/titanic_pipeline.pkl
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.app:app"]
