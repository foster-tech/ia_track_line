# Use a imagem base oficial do Python a partir do Docker Hub
FROM python:3.12-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Instala dependências do sistema para o Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y    

# Install pip and any additional dependencies
RUN pip install --upgrade pip

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .


# Especifica o comando para rodar a aplicação
# Altere "python app.py" para o comando para rodar sua aplicação específica
CMD ["python", "detect-line-track.py"]
