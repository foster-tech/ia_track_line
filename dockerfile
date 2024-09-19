# Use a imagem base oficial do Python a partir do Docker Hub
FROM python:3.12-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Instala dependências do sistema para o Pipenv e Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y    

# Instala o Pipenv
RUN pip install pipenv

# Copia o Pipfile e o Pipfile.lock para o diretório de trabalho
COPY Pipfile Pipfile.lock /app/

# Instala dependências Python em um ambiente virtual criado pelo Pipenv
RUN pipenv install --deploy --system

# Copia o restante do código da aplicação para o diretório de trabalho
COPY . /app

# Especifica o comando para rodar a aplicação
# Altere "python app.py" para o comando para rodar sua aplicação específica
CMD ["python", "detect-line-track.py"]
