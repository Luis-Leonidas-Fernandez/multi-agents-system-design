# Sistema Multi-Agentes con LangGraph

Sistema multi-agentes implementado con LangGraph siguiendo el patrón supervisor.

## 📋 Estructura

```
06_multi_agents/
├── README.md
├── requirements.txt
├── config.py              # Configuración de agentes y modelos
├── agents.py              # Definición de agentes especializados
├── supervisor.py          # Agente supervisor y grafo principal
├── main.py                # Punto de entrada
└── .env.example           # Ejemplo de variables de entorno
```

## 🚀 Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY
```

3. Ejecutar:
```bash
python main.py
```

## 🏗️ Arquitectura

El sistema utiliza un **patrón supervisor** donde:

- **Supervisor**: Coordina y delega tareas a agentes especializados
- **Agente de Matemáticas**: Resuelve problemas matemáticos
- **Agente de Análisis**: Analiza datos y genera reportes
- **Agente de Código**: Escribe y revisa código
- **Agente de Web Scraping**: Extrae información de páginas web

## 📚 Conceptos Clave

- **StateGraph**: Grafo de estado que define el flujo entre agentes
- **Nodes**: Nodos que representan cada agente
- **Edges**: Conexiones que definen cómo fluye la información
- **State**: Estado compartido entre agentes

## 📖 Guías Educativas

Este proyecto incluye guías completas para enseñanza:

### Para Estudiantes:
- **[GUIA_EDUCATIVA.md](./GUIA_EDUCATIVA.md)**: Guía completa paso a paso con explicaciones teóricas y prácticas
- **[CODIGO_PASO_A_PASO.md](./CODIGO_PASO_A_PASO.md)**: Código comentado línea por línea para seguir durante la implementación
- **[DIAGRAMA_FLUJO.md](./DIAGRAMA_FLUJO.md)**: Visualizaciones y diagramas del flujo del sistema

### Para Instructores:
- **[PLAN_CLASE.md](./PLAN_CLASE.md)**: Plan estructurado para una clase de 90 minutos con objetivos, timing y ejercicios

### Orden Recomendado de Estudio:

1. **Leer** `GUIA_EDUCATIVA.md` para entender los conceptos
2. **Seguir** `CODIGO_PASO_A_PASO.md` para implementar paso a paso
3. **Consultar** `DIAGRAMA_FLUJO.md` para visualizar el flujo
4. **Usar** `PLAN_CLASE.md` si estás enseñando

## 🎯 Ejemplos de Uso

- "Calcula la raíz cuadrada de 144"
- "Analiza un dataset de ventas"
- "Escribe una función para calcular factoriales"
- "Extrae información de https://example.com"
- "Scrapea esta página web y dame un resumen"
