# Imagen base de CUDA con soporte para cuDNN y Ubuntu 20.04
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Exponer el puerto que usará Flask
EXPOSE 5000

# Establecer el comando para iniciar la aplicación Flask
CMD ["python3", "servidor.py"]
