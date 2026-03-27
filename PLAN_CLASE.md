# Plan de Clase: Sistema Multi-Agentes con LangGraph

## 🎯 Objetivos de la Clase

1. Comprender el concepto de sistemas multi-agentes
2. Aprender a usar LangGraph para construir agentes
3. Implementar un sistema completo paso a paso
4. Entender el patrón supervisor

---

## ⏱️ Estructura de la Clase (90 minutos)

### Parte 1: Introducción Teórica (15 min)

#### 1.1 ¿Qué son los Sistemas Multi-Agentes? (5 min)
- Definición y conceptos básicos
- Ventajas sobre sistemas de agente único
- Casos de uso reales

#### 1.2 Introducción a LangGraph (5 min)
- ¿Qué es LangGraph?
- Conceptos clave: nodos, edges, estado
- Comparación con otros frameworks

#### 1.3 Patrón Supervisor (5 min)
- Arquitectura del patrón
- Cómo funciona la coordinación
- Ejemplo visual del flujo

---

### Parte 2: Implementación Práctica (60 min)

#### 2.1 Configuración Inicial (10 min)
**Objetivo**: Configurar el proyecto base

**Pasos**:
1. Crear estructura de carpetas
2. Crear `config.py` - Explicar por qué empezar aquí
3. Configurar variables de entorno
4. Crear función `get_llm()`

**Código a escribir**:
```python
# config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    # Implementación
    pass
```

**Preguntas para la clase**:
- ¿Por qué centralizamos la configuración?
- ¿Qué información sensible debemos proteger?

---

#### 2.2 Crear Herramientas (15 min)
**Objetivo**: Entender qué son las herramientas y cómo crearlas

**Pasos**:
1. Explicar el concepto de "tools"
2. Crear herramienta `calculate()`
3. Mostrar cómo el decorador `@tool` funciona
4. Crear más herramientas (analyze_data, write_code)

**Código a escribir**:
```python
# agents.py (primera parte)
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    # Implementación segura
    pass
```

**Conceptos clave**:
- Decorador `@tool`
- Seguridad en evaluación de código
- Manejo de errores

**Ejercicio en clase**: Crear una herramienta `get_time()` que retorne la hora actual

---

#### 2.3 Crear Agentes Especializados (15 min)
**Objetivo**: Crear agentes que usan las herramientas

**Pasos**:
1. Explicar qué es un agente especializado
2. Crear `create_math_agent()`
3. Explicar prompts especializados
4. Crear otros agentes (analysis, code)

**Código a escribir**:
```python
# agents.py (segunda parte)
def create_math_agent():
    llm = get_llm()
    system_prompt = # Crear prompt especializado
    
    # create_react_agent devuelve un grafo compilado listo para usar
    return create_react_agent(
        model=llm,
        tools=[calculate],
        prompt=system_prompt,
        name="math_agent"
    )
```

**Conceptos clave**:
- Prompts especializados (string simple)
- create_react_agent de langgraph.prebuilt
- Integración de herramientas

**Preguntas para la clase**:
- ¿Por qué cada agente tiene un prompt diferente?
- ¿Qué pasa si un agente no tiene las herramientas correctas?

---

#### 2.4 Definir Estado y Crear Nodos (10 min)
**Objetivo**: Entender cómo se comparte información entre agentes

**Pasos**:
1. Explicar el concepto de estado compartido
2. Definir `AgentState` con TypedDict
3. Crear nodos para cada agente
4. Explicar inmutabilidad del estado

**Código a escribir**:
```python
# supervisor.py (primera parte)
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    next_agent: str

def math_node(state: AgentState) -> AgentState:
    # Implementación
    pass
```

**Conceptos clave**:
- Estado compartido
- Inmutabilidad
- Funciones puras

---

#### 2.5 Crear el Supervisor (10 min)
**Objetivo**: Implementar la lógica de decisión

**Pasos**:
1. Explicar el rol del supervisor
2. Crear `supervisor_node()`
3. Implementar lógica de decisión
4. Normalización de respuestas

**Código a escribir**:
```python
# supervisor.py (segunda parte)
def supervisor_node(state: AgentState) -> AgentState:
    # Analizar solicitud
    # Decidir qué agente usar
    # Retornar decisión
    pass
```

**Conceptos clave**:
- Clasificación de tareas
- Toma de decisiones con LLM
- Normalización de respuestas

---

#### 2.6 Construir el Grafo (10 min)
**Objetivo**: Conectar todos los componentes

**Pasos**:
1. Crear StateGraph
2. Agregar todos los nodos
3. Definir edges (condicionales y simples)
4. Compilar el grafo

**Código a escribir**:
```python
# supervisor.py (tercera parte)
def create_supervisor_graph():
    workflow = StateGraph(AgentState)
    # Agregar nodos
    # Definir edges
    # Compilar
    return workflow.compile()
```

**Visualización**: Dibujar el grafo en la pizarra

**Conceptos clave**:
- Entry point
- Conditional edges
- Simple edges
- Compilación

---

### Parte 3: Demostración y Pruebas (10 min)

#### 3.1 Crear Interfaz de Usuario (5 min)
- Crear `main.py`
- Implementar loop interactivo
- Manejo de errores

#### 3.2 Pruebas en Vivo (5 min)
- Probar con diferentes tipos de solicitudes
- Mostrar cómo el supervisor decide
- Demostrar el flujo completo

**Ejemplos a probar**:
1. "Calcula la raíz cuadrada de 144"
2. "Analiza un dataset de ventas"
3. "Escribe una función para calcular factoriales"
4. "Extrae información de https://example.com"

---

### Parte 4: Discusión y Cierre (5 min)

#### 4.1 Preguntas y Respuestas
- ¿Cómo podríamos mejorar el sistema?
- ¿Qué otros agentes podríamos agregar?
- ¿Cómo manejar errores mejor?

#### 4.2 Próximos Pasos
- Ejercicios para casa
- Recursos adicionales
- Proyecto final sugerido

---

## 📝 Materiales Necesarios

### Para el Instructor:
- Código base preparado
- Ejemplos de ejecución
- Diagramas del flujo
- Soluciones a ejercicios

### Para los Estudiantes:
- Computadora con Python 3.10+
- API key de OpenAI (o modelo alternativo)
- Editor de código
- Acceso a internet

---

## 🎯 Puntos Clave a Enfatizar

1. **Modularidad**: Cada componente tiene una responsabilidad clara
2. **Extensibilidad**: Fácil agregar nuevos agentes
3. **Coordinación**: El supervisor es clave para la coordinación
4. **Estado**: Compartir información entre agentes es crucial
5. **Herramientas**: Extienden las capacidades de los agentes

---

## 🏋️ Ejercicios para Después de Clase

### Nivel Básico:
1. Agregar una herramienta `get_date()` que retorne la fecha actual
2. Crear un agente de "traducción" simple
3. Modificar los prompts de los agentes existentes

### Nivel Intermedio:
1. Agregar persistencia de estado con checkpoints
2. Implementar streaming de respuestas
3. Agregar logging y monitoreo

### Nivel Avanzado:
1. Implementar agentes que pueden llamar a otros agentes
2. Agregar un agente de "búsqueda web" real
3. Crear una interfaz web con Streamlit

---

## 📊 Evaluación

### Criterios de Evaluación:
- [ ] Comprensión de conceptos teóricos
- [ ] Implementación correcta de componentes
- [ ] Capacidad de extender el sistema
- [ ] Calidad del código (comentarios, estructura)

### Proyecto Final Sugerido:
Crear un sistema multi-agentes personalizado con:
- Al menos 3 agentes especializados
- 2 herramientas personalizadas
- Interfaz de usuario (CLI o web)
- Documentación completa

---

## 🔗 Recursos Adicionales

- [Guía Educativa Completa](./GUIA_EDUCATIVA.md)
- [Documentación LangGraph](https://docs.langchain.com/oss/python/langgraph/)
- [Ejemplos Oficiales](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Repositorio del Proyecto](https://github.com/tu-usuario/ml-math-from-scratch)

---

## 💡 Tips para el Instructor

1. **Paso a paso**: No apresurarse, asegurarse que todos entiendan cada paso
2. **Visualizaciones**: Usar diagramas para explicar el flujo
3. **Ejemplos prácticos**: Mostrar casos de uso reales
4. **Preguntas frecuentes**: Preparar respuestas a dudas comunes
5. **Debugging**: Mostrar cómo debuggear problemas comunes

---

## ❓ Preguntas Frecuentes

**P: ¿Por qué no usar un solo agente para todo?**
R: Los agentes especializados tienen mejor rendimiento y son más fáciles de mantener.

**P: ¿Cómo sé qué agente usar?**
R: El supervisor decide automáticamente basándose en la solicitud.

**P: ¿Puedo agregar más agentes?**
R: Sí, solo necesitas crear el agente, su nodo, y agregarlo al grafo.

**P: ¿Qué pasa si el supervisor se equivoca?**
R: Puedes mejorar el prompt del supervisor o agregar lógica de validación.

---

¡Buena suerte con tu clase! 🎓
