# Código Paso a Paso: Construyendo el Sistema Multi-Agentes

Esta guía te lleva paso a paso escribiendo el código desde cero. Sigue el orden indicado.

---

## 📁 Paso 1: Crear `config.py`

Crea un nuevo archivo llamado `config.py`:

```python
"""
PASO 1: Configuración del Proyecto
===================================

Este archivo centraliza toda la configuración del proyecto.
Es el primer paso porque todos los demás componentes lo necesitarán.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Cargar variables de entorno desde archivo .env
# Esto permite guardar información sensible (como API keys) fuera del código
load_dotenv()

# Obtener valores del entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Valor por defecto
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

def get_llm():
    """
    Factory function que crea una instancia del modelo LLM.
    
    ¿Por qué una función? Para poder crear múltiples instancias
    cuando sea necesario, todas con la misma configuración.
    """
    # Validar que tenemos la API key
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. "
            "Crea un archivo .env con: OPENAI_API_KEY=tu-key-aqui"
        )
    
    # Crear y retornar el modelo
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
```

**Prueba este paso**:
```python
# En Python o Jupyter
from config import get_llm
llm = get_llm()
print("✅ Configuración correcta!")
```

---

## 📁 Paso 2: Crear `agents.py` - Parte 1 (Herramientas)

Crea un archivo llamado `agents.py` y empieza con las herramientas:

```python
"""
PASO 2: Crear Herramientas (Tools)
===================================

Las herramientas son funciones que los agentes pueden usar
para realizar acciones en el mundo real (más allá de solo generar texto).
"""

from langchain_core.tools import tool
from config import get_llm  # Lo usaremos más adelante

# ==================== HERRAMIENTA 1: CALCULADORA ====================

@tool
def calculate(expression: str) -> str:
    """
    Evalúa una expresión matemática de forma segura.
    
    El decorador @tool convierte esta función en una herramienta
    que los agentes pueden usar automáticamente.
    
    Args:
        expression: Expresión matemática (ej: "2 + 2", "sqrt(16)")
    
    Returns:
        Resultado como string
    """
    try:
        import math
        
        # Crear un namespace seguro
        # Esto previene ejecución de código malicioso
        safe_dict = {
            "__builtins__": {},  # Sin funciones built-in peligrosas
            "math": math,        # Módulo math completo
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        # Evaluar de forma segura
        result = eval(expression, safe_dict)
        return f"Resultado: {result}"
        
    except Exception as e:
        return f"Error al calcular: {str(e)}"


# ==================== HERRAMIENTA 2: ANÁLISIS DE DATOS ====================

@tool
def analyze_data(data_description: str) -> str:
    """
    Analiza datos y genera insights.
    
    En una implementación real, esto podría conectarse a una base de datos
    o procesar archivos. Por ahora, simula el análisis.
    """
    return f"""Análisis de datos: {data_description}

Insights generados:
- Los datos muestran patrones interesantes
- Se recomienda realizar análisis estadístico adicional
- Considerar visualización de los datos para mejor comprensión"""


# ==================== HERRAMIENTA 3: GENERACIÓN DE CÓDIGO ====================

@tool
def write_code(task: str, language: str = "python") -> str:
    """
    Genera código para una tarea específica.
    
    En una implementación real, esto podría usar un modelo de código
    especializado o un sistema de templates.
    """
    return f"""Código {language} para: {task}

```{language}
# Implementación de {task}
def solution():
    # Tu código aquí
    pass
```"""


# ==================== HERRAMIENTA 4: WEB SCRAPING ====================

@tool
def scrape_website(url: str, extract_text: bool = True, extract_links: bool = False) -> str:
    """
    Extrae información de una página web.
    
    Esta herramienta permite a los agentes obtener información
    de internet de forma estructurada.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Headers para simular navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Hacer petición HTTP
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parsear HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        result_parts = [f"URL: {url}\n"]
        
        # Extraer texto si se solicita
        if extract_text:
            # Remover scripts y estilos
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Obtener texto limpio
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limitar tamaño
            if len(text) > 2000:
                text = text[:2000] + "... [texto truncado]"
            
            result_parts.append(f"\nTexto extraído:\n{text}")
        
        # Extraer enlaces si se solicita
        if extract_links:
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text(strip=True)
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                links.append(f"- {link_text}: {href}")
            
            if links:
                links_display = links[:20]  # Limitar a 20
                result_parts.append(f"\n\nEnlaces encontrados:\n")
                result_parts.append('\n'.join(links_display))
        
        return '\n'.join(result_parts)
        
    except Exception as e:
        return f"Error al procesar la página: {str(e)}"
```

**Prueba este paso**:
```python
# Probar las herramientas directamente
from agents import calculate, analyze_data

print(calculate.invoke("2 + 2"))
print(analyze_data.invoke("Dataset de ventas"))
```

---

## 📁 Paso 3: Crear `agents.py` - Parte 2 (Agentes)

Continúa en el mismo archivo `agents.py`:

```python
"""
PASO 3: Crear Agentes Especializados
=====================================

Cada agente usa langgraph.prebuilt.create_react_agent que:
1. Recibe un modelo LLM y herramientas
2. Implementa el patrón ReAct (Reasoning + Acting)
3. Devuelve un grafo compilado listo para usar
"""

from langgraph.prebuilt import create_react_agent

# ==================== AGENTE 1: MATEMÁTICAS ====================

def create_math_agent():
    """
    Crea un agente especializado en matemáticas.
    
    Usa create_react_agent que devuelve un CompiledStateGraph
    que puede ser invocado directamente con mensajes.
    """
    llm = get_llm()  # Obtener modelo desde config
    
    # Prompt del sistema: define la "personalidad" del agente
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


# ==================== AGENTE 2: ANÁLISIS ====================

def create_analysis_agent():
    """Crea un agente especializado en análisis de datos"""
    llm = get_llm()
    
    system_prompt = """Eres un analista de datos experto.
Analizas datos, identificas patrones, y generas insights valiosos.
Proporcionas recomendaciones basadas en evidencia.
Usa la herramienta analyze_data para procesar información."""
    
    return create_react_agent(
        model=llm,
        tools=[analyze_data],
        prompt=system_prompt,
        name="analysis_agent"
    )


# ==================== AGENTE 3: CÓDIGO ====================

def create_code_agent():
    """Crea un agente especializado en programación"""
    llm = get_llm()
    
    system_prompt = """Eres un desarrollador de software experto.
Escribes código limpio, eficiente y bien documentado.
Sigues las mejores prácticas de programación.
Usa la herramienta write_code para generar implementaciones."""
    
    return create_react_agent(
        model=llm,
        tools=[write_code],
        prompt=system_prompt,
        name="code_agent"
    )


# ==================== AGENTE 4: WEB SCRAPING ====================

def create_web_scraping_agent():
    """Crea un agente especializado en web scraping"""
    llm = get_llm()
    
    system_prompt = """Eres un experto en web scraping y extracción de datos web.
Extraes información relevante de páginas web de manera eficiente y ética.
Respetas los robots.txt y términos de servicio.
Usa la herramienta scrape_website para extraer contenido de URLs.
Proporciona resúmenes claros y estructurados de la información extraída."""
    
    return create_react_agent(
        model=llm,
        tools=[scrape_website],
        prompt=system_prompt,
        name="web_scraping_agent"
    )
```

**Prueba este paso**:
```python
# Probar un agente directamente
from agents import create_math_agent
from langchain_core.messages import HumanMessage

math_agent = create_math_agent()
result = math_agent.invoke({"messages": [HumanMessage(content="Calcula 2 + 2")]})
print(result["messages"][-1].content)
```

---

## 📁 Paso 4: Crear `supervisor.py` - Parte 1 (Estado y Nodos)

Crea un archivo llamado `supervisor.py`:

```python
"""
PASO 4: Definir Estado y Crear Nodos
=====================================

El estado es la información compartida entre todos los agentes.
Los nodos son funciones que procesan el estado.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from agents import create_math_agent, create_analysis_agent, create_code_agent, create_web_scraping_agent

# ==================== ESTADO COMPARTIDO ====================

class AgentState(TypedDict):
    """
    Define la estructura del estado compartido.
    
    TypedDict permite que Python entienda la estructura
    y proporcione verificación de tipos.
    """
    # Historial de mensajes: se acumula con cada interacción
    # Annotated permite definir cómo se combinan valores
    messages: Annotated[list, lambda x, y: x + y]
    
    # Próximo agente a ejecutar (decidido por el supervisor)
    next_agent: str


# ==================== CREAR INSTANCIAS DE AGENTES ====================

# Crear una vez, reutilizar muchas veces
math_agent = create_math_agent()
analysis_agent = create_analysis_agent()
code_agent = create_code_agent()
web_scraping_agent = create_web_scraping_agent()


# ==================== NODOS (FUNCIONES DE AGENTES) ====================
# Los agentes creados con create_react_agent reciben y devuelven mensajes

def math_node(state: AgentState) -> AgentState:
    """
    Nodo del agente de matemáticas.
    
    Esta función:
    1. Recibe el estado actual
    2. Extrae el último mensaje
    3. Ejecuta el agente (ahora es un CompiledStateGraph)
    4. Retorna nuevo estado con la respuesta
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


def analysis_node(state: AgentState) -> AgentState:
    """Nodo del agente de análisis"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    result = analysis_agent.invoke({
        "messages": [HumanMessage(content=last_message)]
    })
    
    response_messages = result.get("messages", [])
    if response_messages:
        return {"messages": [AIMessage(content=response_messages[-1].content)]}
    return {"messages": [AIMessage(content="No se pudo procesar.")]}


def code_node(state: AgentState) -> AgentState:
    """Nodo del agente de código"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    result = code_agent.invoke({
        "messages": [HumanMessage(content=last_message)]
    })
    
    response_messages = result.get("messages", [])
    if response_messages:
        return {"messages": [AIMessage(content=response_messages[-1].content)]}
    return {"messages": [AIMessage(content="No se pudo procesar.")]}


def web_scraping_node(state: AgentState) -> AgentState:
    """Nodo del agente de web scraping"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    result = web_scraping_agent.invoke({
        "messages": [HumanMessage(content=last_message)]
    })
    
    response_messages = result.get("messages", [])
    if response_messages:
        return {"messages": [AIMessage(content=response_messages[-1].content)]}
    return {"messages": [AIMessage(content="No se pudo procesar.")]}
```

**Prueba este paso**:
```python
# Probar un nodo directamente
from supervisor import math_node, AgentState
from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="Calcula 2+2")],
    "next_agent": ""
}

result = math_node(state)
print(result["messages"][-1].content)
```

---

## 📁 Paso 5: Crear `supervisor.py` - Parte 2 (Supervisor)

Continúa en `supervisor.py`:

```python
"""
PASO 5: Crear el Supervisor
============================

El supervisor es el "cerebro" que decide qué agente debe
manejar cada solicitud del usuario.
"""

from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

def supervisor_node(state: AgentState) -> AgentState:
    """
    Nodo supervisor que decide qué agente usar.
    
    Proceso:
    1. Analiza la solicitud del usuario
    2. Usa un LLM para decidir qué agente es apropiado
    3. Retorna la decisión en el estado
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
    
    # Extraer el nombre del agente de la respuesta
    agent_name = response.content.strip().lower()
    
    # Normalizar: convertir respuesta del LLM a nombre consistente
    if "math" in agent_name:
        next_agent = "math_agent"
    elif "analysis" in agent_name or "analisis" in agent_name:
        next_agent = "analysis_agent"
    elif "code" in agent_name or "codigo" in agent_name:
        next_agent = "code_agent"
    elif "web" in agent_name or "scraping" in agent_name or "scrape" in agent_name:
        next_agent = "web_scraping_agent"
    else:
        # Fallback: usar math_agent por defecto
        next_agent = "math_agent"
    
    return {"next_agent": next_agent}
```

**Prueba este paso**:
```python
# Probar el supervisor
from supervisor import supervisor_node, AgentState
from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="Calcula 2+2")],
    "next_agent": ""
}

result = supervisor_node(state)
print(f"Supervisor decidió: {result['next_agent']}")
```

---

## 📁 Paso 6: Crear `supervisor.py` - Parte 3 (Enrutamiento y Grafo)

Continúa en `supervisor.py`:

```python
"""
PASO 6: Crear Enrutamiento y Construir el Grafo
===============================================

El enrutamiento decide a dónde ir después de cada nodo.
El grafo conecta todos los componentes.
"""

def route_agent(state: AgentState) -> Literal["math_agent", "analysis_agent", "code_agent", "web_scraping_agent", "supervisor", END]:
    """
    Función de enrutamiento basada en el estado.
    
    Esta función decide a qué nodo ir después basándose en:
    - Si hay mensajes en el estado
    - Qué agente decidió el supervisor
    
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


# ==================== CONSTRUCCIÓN DEL GRAFO ====================

def create_supervisor_graph():
    """
    Crea y retorna el grafo supervisor compilado.
    
    Este es el "cerebro" del sistema que conecta todos los componentes.
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

**Prueba este paso**:
```python
# Probar el grafo completo
from supervisor import create_supervisor_graph
from langchain_core.messages import HumanMessage

graph = create_supervisor_graph()

result = graph.invoke({
    "messages": [HumanMessage(content="Calcula 2+2")],
    "next_agent": ""
})

print(result["messages"][-1].content)
```

---

## 📁 Paso 7: Crear `main.py`

Crea un archivo llamado `main.py`:

```python
"""
PASO 7: Crear Interfaz de Usuario
===================================

Este es el punto de entrada que permite a los usuarios
interactuar con el sistema multi-agentes.
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

**Ejecutar el sistema completo**:
```bash
python main.py
```

---

## ✅ Checklist de Verificación

Después de cada paso, verifica:

- [ ] **Paso 1**: ¿Puedes importar `get_llm()` sin errores?
- [ ] **Paso 2**: ¿Las herramientas se ejecutan correctamente?
- [ ] **Paso 3**: ¿Los agentes responden a preguntas simples?
- [ ] **Paso 4**: ¿Los nodos procesan el estado correctamente?
- [ ] **Paso 5**: ¿El supervisor decide correctamente?
- [ ] **Paso 6**: ¿El grafo se compila sin errores?
- [ ] **Paso 7**: ¿El sistema completo funciona end-to-end?

---

## 🐛 Solución de Problemas Comunes

### Error: "OPENAI_API_KEY no encontrada"
**Solución**: Crea un archivo `.env` con:
```
OPENAI_API_KEY=tu-key-aqui
```

### Error: "ModuleNotFoundError"
**Solución**: Instala dependencias:
```bash
pip install -r requirements.txt
```

### Error: "Agent no responde correctamente"
**Solución**: Verifica que el prompt del agente sea claro y que tenga las herramientas correctas.

### Error: "Supervisor siempre elige el mismo agente"
**Solución**: Mejora el prompt del supervisor con más ejemplos.

---

## 🎓 Conceptos Aprendidos

1. **Configuración centralizada**: Facilita mantenimiento
2. **Herramientas (Tools)**: Extienden capacidades de agentes
3. **Agentes especializados**: Mejor rendimiento en dominios específicos
4. **Estado compartido**: Coordinación entre agentes
5. **Supervisor**: Toma de decisiones inteligente
6. **Grafo de estado**: Visualización y control del flujo
7. **Interfaz de usuario**: Interacción con el sistema

---

¡Felicidades! Has construido un sistema multi-agentes completo desde cero. 🎉
