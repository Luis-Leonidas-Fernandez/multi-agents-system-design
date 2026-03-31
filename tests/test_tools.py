"""
Tests para las tools analyze_data y write_code.
Verifica que produzcan output real en lugar de texto hardcodeado.
"""
import json
import pytest


# ==================== analyze_data ====================

def test_analyze_data_json_array_numerico():
    from agents import analyze_data
    result = analyze_data.func(data="[10, 20, 30, 40, 50]")
    assert "5 valores" in result
    assert "Media" in result
    assert "30" in result          # mediana / media


def test_analyze_data_json_array_stats_correctas():
    from agents import analyze_data
    result = analyze_data.func(data="[2, 4, 6, 8]")
    assert "Mínimo" in result and "2" in result
    assert "Máximo" in result and "8" in result
    assert "5.0" in result or "5" in result   # media = 5


def test_analyze_data_json_tabular():
    from agents import analyze_data
    data = json.dumps([
        {"ventas": 100, "mes": "ene"},
        {"ventas": 200, "mes": "feb"},
        {"ventas": 150, "mes": "mar"},
    ])
    result = analyze_data.func(data=data)
    assert "3 filas" in result
    assert "ventas" in result
    assert "100" in result or "150" in result or "200" in result


def test_analyze_data_csv():
    from agents import analyze_data
    csv_data = "mes,ventas\nene,100\nfeb,200\nmar,150"
    result = analyze_data.func(data=csv_data)
    assert "3 filas" in result
    assert "ventas" in result


def test_analyze_data_fallback_texto():
    from agents import analyze_data
    result = analyze_data.func(data="Tenemos datos de clientes de los últimos 3 meses")
    assert "análisis" in result.lower() or "Datos recibidos" in result
    # No debe ser el stub hardcodeado original
    assert "patrones interesantes" not in result


def test_analyze_data_no_retorna_stub_hardcodeado():
    from agents import analyze_data
    result = analyze_data.func(data="[1, 2, 3]")
    assert "patrones interesantes" not in result
    assert "análisis estadístico adicional" not in result


# ==================== write_code ====================

def test_write_code_python_retorna_codigo_valido():
    from agents import write_code
    result = write_code.func(task="calcular el factorial de un número", language="python")
    assert "```python" in result
    assert "def " in result
    assert "✅" in result           # sintaxis validada


def test_write_code_python_nombre_funcion_generado():
    from agents import write_code
    result = write_code.func(task="ordenar una lista de enteros", language="python")
    assert "def " in result
    assert "pass" not in result     # no debe ser el stub original vacío


def test_write_code_python_docstring():
    from agents import write_code
    result = write_code.func(task="verificar si un número es primo", language="python")
    assert '"""' in result          # docstring presente


def test_write_code_javascript():
    from agents import write_code
    result = write_code.func(task="invertir una cadena", language="javascript")
    assert "```javascript" in result
    assert "function" in result


def test_write_code_typescript():
    from agents import write_code
    result = write_code.func(task="parsear un JSON", language="typescript")
    assert "```typescript" in result


def test_write_code_no_retorna_stub_hardcodeado():
    from agents import write_code
    result = write_code.func(task="mi tarea", language="python")
    # No debe ser el template vacío original
    assert "# Tu código aquí" not in result
    assert "def solution():\n    # Tu código aquí\n    pass" not in result


def test_write_code_go():
    from agents import write_code
    result = write_code.func(task="buscar elemento en slice", language="go")
    assert "```go" in result
    assert "func" in result
