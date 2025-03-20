<p align="center">

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![Code Coverage](coverage.svg)

</p>

# Запуск на своей машине

#### Установка зависимостей
```bash
pip install --user pdm
pdm install
```


Активация окружения
```bash
source .venv/bin/activate
```


Запуск на своей машине
```bash
python -m src.server
```

# Запуск контейнера публично

### Строим контейнер
```bash
sudo docker build -t tool_backend .
```
Узнаем его IMAGE ID 
```bash
sudo docker images
```

```bash
docker run -d -p 7070:7070 bb1942a77c32
```

```bash
docker run -d -p 80:7070 bb1942a77c32
```

```bash
docker run -d -p 7070:80 <IMAGE ID>
```



# Запуск контейнера локально

### Строим контейнер
```bash
sudo docker build -t tool_backend .
```
Узнаем его ID
```bash
sudo docker images
```

```bash
sudo docker run -d -p 7070:7071 --cpuset-cpus="0-10" --memory="42g" bbb1920d4ef5
```