FROM python:3.9.16-slim-buster


COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

ENV PYTHONPATH=/app

COPY pyproject.toml .

RUN pip install -U pip setuptools wheel
RUN pip install zstandard
RUN pip install pdm
RUN pdm install --prod --frozen-lockfile --no-editable

EXPOSE 7078

ENTRYPOINT ["pdm", "run", "src/server.py"]

