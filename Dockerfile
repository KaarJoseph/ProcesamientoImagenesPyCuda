# Imagen base con CUDA y Python
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

# Instalación de paquetes necesarios
RUN apt-get -qq update && \
    apt-get -qq install -y build-essential python3-pip && \
    pip3 install pycuda flask numpy pillow

# Copiar el código de la aplicación
COPY . /app
WORKDIR /app

# Exponer el puerto de Flask
EXPOSE 5000

# Comando para iniciar Flask
CMD ["python3", "servidor.py"]
