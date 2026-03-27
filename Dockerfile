# syntax=docker/dockerfile:1

# ============================================
# Base: imagen oficial de Playwright con Python
# Incluye Chromium, Firefox y WebKit preinstalados
# ============================================
FROM mcr.microsoft.com/playwright/python:v1.57.0-jammy

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Comando por defecto
CMD ["python", "main.py"]
