# Guía Educativa: Construyendo un Sistema Multi-Agentes con LangGraph

## 📚 Objetivos de Aprendizaje

Al finalizar esta guía, serás capaz de:
- Entender el patrón supervisor en sistemas multi-agentes
- Implementar agentes especializados con LangGraph
- Crear herramientas (tools) para agentes
- Construir un grafo de estado con LangGraph
- Coordinar múltiples agentes trabajando juntos

---

## 🎯 Conceptos Teóricos

### ¿Qué es un Sistema Multi-Agentes?

Un sistema multi-agentes es un conjunto de agentes autónomos que trabajan juntos para resolver problemas complejos. Cada agente tiene:
- **Especialización**: Conocimiento en un dominio específico
- **Autonomía**: Puede tomar decisiones
- **Comunicación**: Puede interactuar con otros agentes

### Patrón Supervisor

El patrón supervisor es una arquitectura donde:
- Un **supervisor** coordina y delega tareas
- **Agentes especializados** ejecutan tareas específicas
- El supervisor decide qué agente debe manejar cada solicitud

### LangGraph

LangGraph es un framework que permite construir agentes como **grafos de estado**:
- **Nodos**: Representan agentes o funciones
- **Edges**: Representan el flujo de información
- **Estado**: Información compartida entre nodos

---

## 🛠️ Construcción Paso a Paso

### Paso 1: Configuración del Proyecto

**Archivo: `config.py`**

**Propósito**: Centralizar la configuración del proyecto, especialmente la conexión al modelo LLM.

**¿Por qué empezar aquí?**
- Todos los agentes necesitan acceso al modelo LLM
- Es más fácil mantener la configuración en un solo lugar
- Facilita cambiar de modelo sin modificar múltiples archivos

```python
"""
config.py - Configuración centralizada del proyecto
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Obtener configuración del entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Modelo por defecto
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))    # Creatividad del modelo

def get_llm():
    """
    Crea y retorna una instancia del modelo LLM.
    
    Esta función centraliza la creación del modelo para que todos los agentes
    usen la misma configuración.
    
    Returns:
        ChatOpenAI: Instancia del modelo configurado
    
    Raises:
        ValueError: Si no se encuentra la API key
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. "
            "Por favor configura tu API key en el archivo .env"
        )
    
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
```

**Conceptos clave**:
- **Variables de entorno**: Almacenan información sensible (API keys)
- **Función factory**: `get_llm()` crea instancias del modelo cuando se necesitan
- **Configuración centralizada**: Un solo lugar para cambiar el modelo

---

### Paso 2: Crear Herramientas (Tools)

**Archivo: `agents.py` (primera parte)**

**Propósito**: Definir las herramientas que los agentes pueden usar para realizar acciones.

**¿Por qué crear herramientas primero?**
- Los agentes necesitan herramientas para interactuar con el mundo
- Las herramientas encapsulan funcionalidades específicas
- Permiten que los agentes realicen acciones más allá de solo generar texto

```python
"""
agents.py - Definición de herramientas y agentes especializados
"""
from langchain_core.tools import tool
from config import get_llm

# ==================== HERRAMIENTAS ====================

@tool
def calculate(expression: str) -> str:
    """
    Evalúa una expresión matemática de forma segura.
    
    Esta herramienta permite a los agentes realizar cálculos matemáticos
    sin tener que confiar en el LLM para hacer aritmética.
    
    Args:
        expression: Expresión matemática (ej: "2 + 2", "sqrt(16)")
    
    Returns:
        Resultado de la expresión como string
    """
    try:
        import math
        # Namespace seguro: solo permite funciones matemáticas
        # Esto previene ejecución de código malicioso
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, safe_dict)
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"
```

**Conceptos clave**:
- **Decorador `@tool`**: Convierte una función en una herramienta que los agentes pueden usar
- **Namespace seguro**: Limita qué funciones pueden ejecutarse (seguridad)
- **Manejo de errores**: Retorna mensajes de error útiles

**Ejercicio**: Agrega más herramientas siguiendo este patrón:
- `analyze_data()`: Para análisis de datos
- `write_code()`: Para generar código
- `scrape_website()`: Para web scraping

---

### Paso 3: Crear Agentes Especializados

**Archivo: `agents.py` (segunda parte)**

**Propósito**: Crear agentes que usan las herramientas para resolver tareas específicas.

**¿Por qué especializar agentes?**
- Cada agente se enfoca en un dominio específico
- Mejor rendimiento al tener prompts especializados
- Más fácil de mantener y mejorar

```python
from langgraph.prebuilt import create_react_agent

def create_math_agent():
    """
    Crea un agente especializado en matemáticas.
    
    Usa create_react_agent que devuelve un CompiledStateGraph
    que implementa el patrón ReAct (Reasoning + Acting).
    
    Returns:
        CompiledStateGraph: Agente listo para usar
    """
    llm = get_llm()  # Obtener el modelo desde config
    
    # Prompt del sistema: define la personalidad y capacidades del agente
    system_prompt = """Eres un experto matemático. 
Resuelves problemas matemáticos complejos, desde álgebra hasta cálculo.
Usa la herramienta calculate para evaluar expresiones matemáticas.
Sé preciso y muestra tu razonamiento paso a paso."""
    
    # create_react_agent devuelve un grafo compilado listo para usar
    return create_react_agent(
        model=llm,
        tools=[calculate],
        prompt=system_prompt,
        name="math_agent"
    )
```

**Conceptos clave**:
- **create_react_agent**: Crea un agente que sigue el patrón ReAct
- **Prompt como string**: Define el rol y comportamiento del agente
- **CompiledStateGraph**: El agente es un grafo que recibe y devuelve mensajes

**Ejercicio**: Crea agentes similares para:
- Análisis de datos
- Programación
- Web scraping

---

### Paso 4: Definir el Estado Compartido

**Archivo: `supervisor.py` (primera parte)**

**Propósito**: Definir qué información se comparte entre todos los agentes.

**¿Por qué necesitamos estado compartido?**
- Los agentes necesitan ver el historial de conversación
- El supervisor necesita saber qué agente debe ejecutarse
- Permite coordinar múltiples agentes

```python
"""
supervisor.py - Sistema supervisor multi-agentes
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# ==================== ESTADO COMPARTIDO ====================

class AgentState(TypedDict):
    """
    Estado compartido entre todos los agentes.
    
    TypedDict permite que Python entienda la estructura del estado
    y proporcione autocompletado y verificación de tipos.
    """
    # Historial de mensajes: se acumula con cada interacción
    messages: Annotated[list, lambda x, y: x + y]  
    
    # Próximo agente a ejecutar (decidido por el supervisor)
    next_agent: str
```

**Conceptos clave**:
- **TypedDict**: Define la estructura del estado con tipos
- **Annotated**: Especifica cómo se combinan valores (x + y para listas)
- **Estado inmutable**: Cada nodo retorna un nuevo estado, no modifica el existente

---

### Paso 5: Crear Nodos (Funciones de Agentes)

**Archivo: `supervisor.py` (segunda parte)**

**Propósito**: Crear funciones que representan cada agente en el grafo.

**¿Por qué nodos separados?**
- Cada nodo es independiente y testeable
- Facilita agregar o quitar agentes
- Permite paralelización futura

```python
from agents import create_math_agent, create_analysis_agent, create_code_agent

# Crear instancias de los agentes (una vez, reutilizables)
math_agent = create_math_agent()
analysis_agent = create_analysis_agent()
code_agent = create_code_agent()

def math_node(state: AgentState) -> AgentState:
    """
    Nodo del agente de matemáticas.
    
    Esta función:
    1. Recibe el estado actual
    2. Extrae el último mensaje
    3. Ejecuta el agente (ahora es un CompiledStateGraph)
    4. Retorna el nuevo estado con la respuesta
    
    Args:
        state: Estado actual del sistema
    
    Returns:
        Nuevo estado con la respuesta del agente
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Los agentes create_react_agent reciben mensajes
    result = math_agent.invoke({
        "messages": [HumanMessage(content=last_message)]
    })
    
    # Obtener la respuesta del último mensaje
    response_messages = result.get("messages", [])
    if response_messages:
        return {"messages": [AIMessage(content=response_messages[-1].content)]}
    return {"messages": [AIMessage(content="No se pudo procesar.")]}
```

**Conceptos clave**:
- **Función pura**: Recibe estado, retorna nuevo estado (sin efectos secundarios)
- **Mensajes**: Los agentes ReAct reciben y devuelven mensajes
- **Inmutabilidad**: No modifica el estado original, crea uno nuevo

**Ejercicio**: Crea nodos similares para otros agentes.

---

### Paso 6: Crear el Supervisor

**Archivo: `supervisor.py` (tercera parte)**

**Propósito**: Crear el agente que decide qué agente especializado debe manejar cada tarea.

**¿Por qué necesitamos un supervisor?**
- El usuario no sabe qué agente usar
- El supervisor analiza la solicitud y decide automáticamente
- Permite manejar tareas complejas que requieren múltiples agentes

```python
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

def supervisor_node(state: AgentState) -> AgentState:
    """
    Nodo supervisor que decide qué agente usar.
    
    Este es el "cerebro" del sistema que:
    1. Analiza la solicitud del usuario
    2. Decide qué agente especializado debe manejarla
    3. Actualiza el estado con su decisión
    
    Args:
        state: Estado actual con los mensajes del usuario
    
    Returns:
        Estado actualizado con next_agent definido
    """
    llm = get_llm()
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Prompt que enseña al supervisor a decidir
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un supervisor que coordina un equipo de agentes especializados.
        
        Tienes acceso a estos agentes:
        1. math_agent: Para problemas matemáticos, cálculos, álgebra, etc.
        2. analysis_agent: Para análisis de datos, estadísticas, insights
        3. code_agent: Para escribir código, programación, desarrollo
        4. web_scraping_agent: Para extraer información de páginas web
        
        Tu trabajo es analizar la solicitud y decidir qué agente debe manejarla.
        Responde SOLO con el nombre del agente: "math_agent", "analysis_agent", 
        "code_agent", o "web_scraping_agent".
        
        Ejemplos:
        - "Calcula 2+2" -> math_agent
        - "Analiza estos datos" -> analysis_agent
        - "Escribe código para..." -> code_agent
        - "Extrae info de https://..." -> web_scraping_agent"""),
        ("user", "Solicitud: {input}\n\n¿Qué agente debe manejar esto?"),
    ])
    
    # Ejecutar el LLM para obtener la decisión
    chain = prompt | llm
    response = chain.invoke({"input": last_message})
    
    # Extraer y normalizar el nombre del agente
    agent_name = response.content.strip().lower()
    
    # Lógica de decisión (normalización)
    if "math" in agent_name:
        next_agent = "math_agent"
    elif "analysis" in agent_name or "analisis" in agent_name:
        next_agent = "analysis_agent"
    elif "code" in agent_name or "codigo" in agent_name:
        next_agent = "code_agent"
    elif "web" in agent_name or "scraping" in agent_name or "scrape" in agent_name:
        next_agent = "web_scraping_agent"
    else:
        next_agent = "math_agent"  # Por defecto
    
    return {"next_agent": next_agent}
```

**Conceptos clave**:
- **Clasificación de tareas**: El supervisor clasifica la solicitud
- **Normalización**: Convierte respuestas del LLM a nombres consistentes
- **Fallback**: Tiene un agente por defecto si no está seguro

---

### Paso 7: Crear la Función de Enrutamiento

**Archivo: `supervisor.py` (cuarta parte)**

**Propósito**: Decidir a qué nodo ir basándose en el estado actual.

**¿Por qué necesitamos enrutamiento?**
- El grafo necesita saber a dónde ir después de cada nodo
- Permite flujos condicionales (no siempre lineal)
- Controla el flujo de ejecución

```python
def route_agent(state: AgentState) -> Literal["math_agent", "analysis_agent", "code_agent", "web_scraping_agent", "supervisor", END]:
    """
    Función de enrutamiento basada en el estado.
    
    Esta función decide a qué nodo ir después basándose en:
    - Si hay mensajes en el estado
    - Qué agente decidió el supervisor
    
    Args:
        state: Estado actual del sistema
    
    Returns:
        Nombre del próximo nodo a ejecutar
    """
    # Si no hay mensajes, terminar
    if not state.get("messages"):
        return END
    
    # Obtener la decisión del supervisor
    next_agent = state.get("next_agent")
    
    # Enrutar al agente correspondiente
    if next_agent == "math_agent":
        return "math_agent"
    elif next_agent == "analysis_agent":
        return "analysis_agent"
    elif next_agent == "code_agent":
        return "code_agent"
    elif next_agent == "web_scraping_agent":
        return "web_scraping_agent"
    else:
        # Si no hay decisión, ir al supervisor primero
        return "supervisor"
```

**Conceptos clave**:
- **Conditional edges**: Permiten flujos no lineales
- **Literal types**: TypeScript/Python ayuda a verificar que los valores sean válidos
- **END**: Nodo especial que termina la ejecución

---

### Paso 8: Construir el Grafo

**Archivo: `supervisor.py` (quinta parte)**

**Propósito**: Conectar todos los nodos para crear el sistema completo.

**¿Por qué usar un grafo?**
- Visualiza el flujo del sistema
- Facilita agregar/quitar agentes
- Permite flujos complejos (ciclos, ramificaciones)

```python
def create_supervisor_graph():
    """
    Crea y retorna el grafo supervisor compilado.
    
    Este es el "cerebro" del sistema que conecta todos los componentes:
    1. Define todos los nodos (agentes)
    2. Define cómo fluye la información (edges)
    3. Compila el grafo para ejecución
    
    Returns:
        Grafo compilado listo para usar
    """
    # Crear un grafo de estado vacío
    workflow = StateGraph(AgentState)
    
    # ========== AGREGAR NODOS ==========
    # Cada nodo es una función que procesa el estado
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("math_agent", math_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("web_scraping_agent", web_scraping_node)
    
    # ========== DEFINIR FLUJO ==========
    # Punto de entrada: siempre empieza en el supervisor
    workflow.set_entry_point("supervisor")
    
    # Edge condicional: el supervisor decide a dónde ir
    workflow.add_conditional_edges(
        "supervisor",           # Nodo origen
        route_agent,            # Función que decide
        {
            "math_agent": "math_agent",
            "analysis_agent": "analysis_agent",
            "code_agent": "code_agent",
            "web_scraping_agent": "web_scraping_agent",
            END: END
        }
    )
    
    # Edges simples: todos los agentes vuelven al supervisor
    # Esto permite que el supervisor decida el siguiente paso
    workflow.add_edge("math_agent", "supervisor")
    workflow.add_edge("analysis_agent", "supervisor")
    workflow.add_edge("code_agent", "supervisor")
    workflow.add_edge("web_scraping_agent", "supervisor")
    
    # Compilar el grafo (prepararlo para ejecución)
    return workflow.compile()
```

**Conceptos clave**:
- **Entry point**: Dónde empieza la ejecución
- **Conditional edges**: Flujos que dependen de condiciones
- **Simple edges**: Flujos fijos (siempre van de A a B)
- **Compile**: Prepara el grafo para ejecución eficiente

**Visualización del grafo**:
```
                    [START]
                       ↓
                  [supervisor]
                       ↓
            ┌──────────┼──────────┐
            ↓          ↓          ↓
      [math_agent] [analysis] [code_agent] [web_scraping]
            ↓          ↓          ↓          ↓
            └──────────┴──────────┴──────────┘
                       ↓
                  [supervisor]
                       ↓
                    [END]
```

---

### Paso 9: Crear el Punto de Entrada

**Archivo: `main.py`**

**Propósito**: Interfaz de usuario para interactuar con el sistema.

**¿Por qué un archivo main separado?**
- Separa la lógica del sistema de la interfaz
- Facilita crear diferentes interfaces (CLI, web, API)
- Permite reutilizar el grafo en diferentes contextos

```python
"""
main.py - Punto de entrada principal
"""
from supervisor import create_supervisor_graph
from langchain_core.messages import HumanMessage

def main():
    """Función principal que ejecuta el sistema interactivo"""
    
    # Mostrar información al usuario
    print("=" * 60)
    print("Sistema Multi-Agentes con LangGraph")
    print("=" * 60)
    print("\nAgentes disponibles:")
    print("  - math_agent: Problemas matemáticos")
    print("  - analysis_agent: Análisis de datos")
    print("  - code_agent: Programación y código")
    print("  - web_scraping_agent: Extracción de información web")
    print("\nEscribe 'salir' para terminar\n")
    
    # Crear el grafo (una vez al inicio)
    graph = create_supervisor_graph()
    
    # Estado inicial (vacío)
    state = {
        "messages": [],
        "next_agent": ""
    }
    
    # Loop principal de interacción
    while True:
        # Obtener input del usuario
        user_input = input("\nTu pregunta: ").strip()
        
        # Salir si el usuario lo solicita
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\n¡Hasta luego!")
            break
        
        if not user_input:
            continue
        
        # Agregar mensaje del usuario al estado
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Ejecutar el grafo con el estado actual
            print("\n🤖 Procesando...\n")
            result = graph.invoke(state)
            
            # Mostrar respuesta
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"\n📝 Respuesta:\n{last_message.content}\n")
            
            # Actualizar estado para la próxima iteración
            state = result
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            print("Por favor verifica tu configuración (API key, etc.)")

if __name__ == "__main__":
    main()
```

**Conceptos clave**:
- **Loop interactivo**: Permite múltiples consultas
- **Manejo de estado**: Mantiene el contexto entre interacciones
- **Manejo de errores**: Informa al usuario si algo falla

---

## 📋 Checklist de Implementación

Sigue este orden para construir el sistema:

- [ ] **Paso 1**: Crear `config.py` con función `get_llm()`
- [ ] **Paso 2**: Crear herramientas en `agents.py` (calculate, analyze_data, etc.)
- [ ] **Paso 3**: Crear agentes especializados en `agents.py`
- [ ] **Paso 4**: Definir `AgentState` en `supervisor.py`
- [ ] **Paso 5**: Crear nodos para cada agente en `supervisor.py`
- [ ] **Paso 6**: Crear nodo supervisor en `supervisor.py`
- [ ] **Paso 7**: Crear función de enrutamiento en `supervisor.py`
- [ ] **Paso 8**: Construir el grafo en `supervisor.py`
- [ ] **Paso 9**: Crear `main.py` para interactuar con el sistema

---

## 🧪 Ejercicios Prácticos

### Ejercicio 1: Agregar una Nueva Herramienta
Crea una herramienta `get_weather(city: str)` que simule obtener el clima de una ciudad.

### Ejercicio 2: Crear un Nuevo Agente
Crea un agente de "traducción" que traduzca texto entre idiomas.

### Ejercicio 3: Modificar el Flujo
Cambia el flujo para que después de web_scraping, siempre vaya al analysis_agent.

### Ejercicio 4: Agregar Memoria Persistente
Investiga cómo agregar checkpoints a LangGraph para que el sistema recuerde conversaciones anteriores.

---

## 🎓 Preguntas de Reflexión

1. **¿Por qué es mejor tener agentes especializados en lugar de un solo agente?**
   - Mejor rendimiento en tareas específicas
   - Más fácil de mantener y mejorar
   - Permite paralelización

2. **¿Qué pasaría si elimináramos el supervisor?**
   - El usuario tendría que saber qué agente usar
   - No habría coordinación entre agentes
   - Tareas complejas serían más difíciles

3. **¿Cómo podríamos mejorar el sistema?**
   - Agregar persistencia de estado
   - Implementar streaming de respuestas
   - Agregar más agentes especializados
   - Mejorar el manejo de errores

---

## 📚 Recursos Adicionales

- [Documentación oficial de LangGraph](https://docs.langchain.com/oss/python/langgraph/)
- [Ejemplos de LangGraph](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Patrón Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

---

## ✅ Resumen

Has aprendido a construir un sistema multi-agentes completo:

1. **Configuración centralizada** → Facilita el mantenimiento
2. **Herramientas especializadas** → Extienden las capacidades de los agentes
3. **Agentes especializados** → Mejor rendimiento en dominios específicos
4. **Estado compartido** → Coordinación entre agentes
5. **Supervisor inteligente** → Decisión automática de qué agente usar
6. **Grafo de estado** → Visualización y control del flujo
7. **Interfaz de usuario** → Interacción con el sistema

¡Felicidades! Ahora tienes un sistema multi-agentes funcional y extensible.
