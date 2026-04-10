# Diagrama de Flujo del Sistema Multi-Agentes

## 🎨 Visualización del Sistema

### Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                    USUARIO                                    │
│              (Hace una pregunta)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  ESTADO COMPARTIDO                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  messages: [Historial de conversación]            │    │
│  │  next_agent: "math_agent"                         │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Analiza la solicitud del usuario               │    │
│  │  2. Usa LLM para decidir qué agente usar           │    │
│  │  3. Retorna: next_agent = "math_agent"             │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ↓              ↓              ↓
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│   MATH    │  │ ANALYSIS  │  │   CODE    │  │   WEB     │
│   AGENT   │  │   AGENT   │  │   AGENT   │  │ SCRAPING  │
│           │  │           │  │           │  │   AGENT   │
│ Usa:      │  │ Usa:      │  │ Usa:      │  │ Usa:      │
│ calculate │  │analyze_   │  │write_code │  │scrape_    │
│           │  │data       │  │           │  │website     │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                       │
                       ↓
              ┌────────────────┐
              │  SUPERVISOR    │
              │  (Decide si    │
              │   continuar)   │
              └────────┬───────┘
                       │
                       ↓
              ┌────────────────┐
              │  RESPUESTA     │
              │  AL USUARIO    │
              └────────────────┘
```

---

## 🔄 Flujo Detallado Paso a Paso

### Ejemplo: "Calcula la raíz cuadrada de 144"

```
PASO 1: Usuario envía solicitud
────────────────────────────────
Usuario: "Calcula la raíz cuadrada de 144"
         ↓
Estado inicial:
  messages: [HumanMessage("Calcula la raíz cuadrada de 144")]
  next_agent: ""


PASO 2: Supervisor analiza
──────────────────────────
Supervisor recibe estado
         ↓
Analiza con LLM: "¿Qué agente debe manejar esto?"
         ↓
LLM responde: "math_agent"
         ↓
Estado actualizado:
  messages: [HumanMessage("Calcula la raíz cuadrada de 144")]
  next_agent: "math_agent"


PASO 3: Router decide
─────────────────────
route_agent() lee next_agent = "math_agent"
         ↓
Enruta a: math_node


PASO 4: Math Agent procesa
──────────────────────────
math_node ejecuta:
         ↓
math_agent.invoke({
  "input": "Calcula la raíz cuadrada de 144"
})
         ↓
Agente razona: "Necesito usar calculate('sqrt(144)')"
         ↓
Llama a calculate("sqrt(144)")
         ↓
calculate retorna: "Resultado: 12.0"
         ↓
Agente genera respuesta final
         ↓
Estado actualizado:
  messages: [
    HumanMessage("Calcula la raíz cuadrada de 144"),
    AIMessage("La raíz cuadrada de 144 es 12.0")
  ]
  next_agent: "math_agent"


PASO 5: Vuelve al supervisor
────────────────────────────
math_node → supervisor (edge simple)
         ↓
Supervisor decide: ¿Hay más trabajo?
         ↓
En este caso: No, terminar
         ↓
Estado final:
  messages: [
    HumanMessage("Calcula la raíz cuadrada de 144"),
    AIMessage("La raíz cuadrada de 144 es 12.0")
  ]
  next_agent: ""


PASO 6: Respuesta al usuario
────────────────────────────
Sistema retorna el último mensaje:
  "La raíz cuadrada de 144 es 12.0"
```

---

## 📊 Estructura del Grafo

### Nodos y Conexiones

```
                    [START]
                       │
                       ↓
                ┌──────────────┐
                │  SUPERVISOR  │
                │   (Nodo)     │
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ↓              ↓              ↓
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  MATH    │   │ANALYSIS │   │  CODE   │   │   WEB   │
   │  NODE    │   │  NODE   │   │  NODE   │   │  NODE   │
   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                       │
                       ↓
                ┌──────────────┐
                │  SUPERVISOR  │
                │   (Nuevamente)│
                └──────┬───────┘
                       │
                  ┌────┴────┐
                  │   END   │
                  └─────────┘
```

### Tipos de Edges

1. **Entry Point**: `START → supervisor`
   - Siempre empieza aquí

2. **Conditional Edge**: `supervisor → [math|analysis|code|web]`
   - Depende de la decisión del supervisor

3. **Simple Edge**: `[math|analysis|code|web] → supervisor`
   - Siempre vuelve al supervisor

---

## 🔧 Componentes del Sistema

### 1. Estado (AgentState)

```
┌─────────────────────────────────────┐
│         AgentState                  │
├─────────────────────────────────────┤
│ messages: List[Message]             │
│   ├─ HumanMessage                   │
│   ├─ AIMessage                      │
│   └─ ...                            │
│                                     │
│ next_agent: str                     │
│   └─ "math_agent" | "analysis_..." │
└─────────────────────────────────────┘
```

### 2. Supervisor

```
┌─────────────────────────────────────┐
│      supervisor_node()              │
├─────────────────────────────────────┤
│ Input:  AgentState                   │
│         └─ messages: [solicitud]     │
│                                     │
│ Proceso:                             │
│   1. Extrae último mensaje          │
│   2. LLM analiza la solicitud        │
│   3. LLM decide qué agente usar      │
│   4. Normaliza respuesta             │
│                                     │
│ Output: AgentState                   │
│         └─ next_agent: "math_agent" │
└─────────────────────────────────────┘
```

### 3. Agente Especializado (Ejemplo: Math Agent)

```
┌─────────────────────────────────────┐
│         math_node()                  │
├─────────────────────────────────────┤
│ Input:  AgentState                   │
│         └─ messages: [solicitud]    │
│                                     │
│ Proceso:                             │
│   1. Extrae último mensaje           │
│   2. Ejecuta math_agent              │
│      ├─ Agente razona                │
│      ├─ Decide usar calculate()     │
│      ├─ Llama a calculate()          │
│      └─ Genera respuesta             │
│                                     │
│ Output: AgentState                   │
│         └─ messages: [respuesta]     │
└─────────────────────────────────────┘
```

---

## 🎯 Ejemplos de Flujos

### Flujo Simple (1 agente)

```
Usuario: "Calcula 2+2"
    ↓
Supervisor → math_agent
    ↓
Respuesta: "4"
    ↓
END
```

### Flujo Complejo (múltiples agentes)

```
Usuario: "Extrae info de https://example.com y analízala"
    ↓
Supervisor → web_scraping_agent
    ↓
web_scraping_agent: Extrae información
    ↓
Supervisor → analysis_agent
    ↓
analysis_agent: Analiza la información extraída
    ↓
Respuesta combinada
    ↓
END
```

---

## 📈 Visualización del Estado

### Estado Inicial
```
┌─────────────────────────────────────┐
│ messages: []                        │
│ next_agent: ""                      │
└─────────────────────────────────────┘
```

### Después de Solicitud del Usuario
```
┌─────────────────────────────────────┐
│ messages: [                          │
│   HumanMessage("Calcula 2+2")        │
│ ]                                    │
│ next_agent: ""                       │
└─────────────────────────────────────┘
```

### Después de Supervisor
```
┌─────────────────────────────────────┐
│ messages: [                          │
│   HumanMessage("Calcula 2+2")        │
│ ]                                    │
│ next_agent: "math_agent"             │
└─────────────────────────────────────┘
```

### Después de Math Agent
```
┌─────────────────────────────────────┐
│ messages: [                          │
│   HumanMessage("Calcula 2+2"),       │
│   AIMessage("El resultado es 4")    │
│ ]                                    │
│ next_agent: "math_agent"             │
└─────────────────────────────────────┘
```

---

## 🔍 Puntos Clave del Flujo

1. **Todo empieza en el supervisor**
   - El supervisor es el punto de entrada

2. **El supervisor decide, no ejecuta**
   - Solo decide qué agente usar
   - No procesa la tarea directamente

3. **Los agentes especializados ejecutan**
   - Cada agente procesa su dominio
   - Usan herramientas para realizar acciones

4. **Siempre se vuelve al supervisor**
   - Permite coordinar múltiples pasos
   - Permite manejar tareas complejas

5. **El estado se acumula**
   - Los mensajes se van agregando
   - Mantiene el contexto completo

---

## 🎓 Para la Clase

### Diagrama Simplificado para Pizarra

```
Usuario
   │
   ↓
[Supervisor] ←──┐
   │            │
   ├─→ [Math] ──┤
   ├─→ [Analysis] ──┤
   ├─→ [Code] ──┤
   └─→ [Web] ───┘
```

### Explicación Oral

1. "El usuario hace una pregunta"
2. "El supervisor la analiza y decide qué agente usar"
3. "El agente especializado procesa la tarea usando sus herramientas"
4. "El agente retorna al supervisor"
5. "El supervisor decide si hay más trabajo o terminar"
6. "Se retorna la respuesta al usuario"

---

¡Usa estos diagramas para explicar el sistema en tu clase! 🎓
