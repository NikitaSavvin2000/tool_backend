FROM python:3.11-buster
USER root
RUN apt-get update
RUN apt-get install -y vim poppler-utils

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .

RUN pip install -U pip setuptools wheel
RUN pip install zstandard
RUN pip install pdm
RUN pdm install --prod --frozen-lockfile --no-editable

EXPOSE 7079

ENTRYPOINT ["pdm", "run", "src/server.py"]

