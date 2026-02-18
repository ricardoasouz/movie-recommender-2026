FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY packages/recommender/ /app/packages/recommender/

RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir \
     pandas numpy scipy scikit-learn joblib \
  && pip install --no-cache-dir -e /app/packages/recommender

WORKDIR /app

ENTRYPOINT ["python", "-m", "recommender.build_index_cf"]
