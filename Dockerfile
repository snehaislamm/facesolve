# Reproducible training/eval environment: same container on a workstation or a cloud
# GPU/CPU instance (e.g. AWS EC2 or SageMaker training job).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY facesolve/ facesolve/
COPY scripts/ scripts/
COPY tests/ tests/

# fail the build if the code is broken
RUN python -m pytest -q tests

# results/ is a volume so the trained model and .usda land on the host
VOLUME ["/app/results"]
CMD ["python", "scripts/run_experiment.py"]
