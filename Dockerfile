FROM python:3.9.16-slim-buster

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .
COPY pdm.lock .

RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pdm install --prod --no-lock --no-editable

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7070/docs || exit 1

EXPOSE 7070

ENTRYPOINT ["pdm", "run", "src/server.py"]
