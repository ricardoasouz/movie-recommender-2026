FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy API service and recommender package into the image
COPY services/api/ /app/services/api/
COPY packages/recommender/ /app/packages/recommender/

# Install runtime deps + install local package (editable not required inside container, but ok)
RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir \
     fastapi==0.115.* uvicorn[standard]==0.30.* \
     psycopg[binary]==3.* sqlmodel==0.0.* sqlalchemy==2.* \
     pandas numpy scipy scikit-learn joblib \
  && pip install --no-cache-dir -e /app/packages/recommender

WORKDIR /app/services/api

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
