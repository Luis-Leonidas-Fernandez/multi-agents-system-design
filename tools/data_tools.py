"""Tools de análisis de datos del sistema."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def analyze_data(
    data: Annotated[str, Field(
        description=(
            "Datos a analizar. Puede ser: "
            "(1) JSON array de números, ej: [10, 20, 30], "
            "(2) JSON array de objetos, ej: [{\"ventas\": 100, \"mes\": \"ene\"}, ...], "
            "(3) CSV con encabezado, ej: 'mes,ventas\\nene,100\\nfeb,200', "
            "(4) descripción textual si no hay datos estructurados."
        )
    )],
) -> str:
    """
    Analiza datos estructurados (JSON, CSV) y retorna estadísticas reales.
    Si los datos no son estructurados, retorna un framework de análisis.
    """
    import json
    import statistics
    import csv
    import io

    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) for x in parsed):
            n = len(parsed)
            mean = statistics.mean(parsed)
            return (
                f"Análisis estadístico — {n} valores numéricos:\n"
                f"- Mínimo:              {min(parsed)}\n"
                f"- Máximo:              {max(parsed)}\n"
                f"- Media:               {mean:.4f}\n"
                f"- Mediana:             {statistics.median(parsed):.4f}\n"
                f"- Desviación estándar: {statistics.stdev(parsed):.4f}\n"
                f"- Suma:                {sum(parsed)}\n"
                f"- Rango:               {max(parsed) - min(parsed)}"
            )
    except (json.JSONDecodeError, TypeError, statistics.StatisticsError):
        pass

    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            columns = list(parsed[0].keys())
            lines = [f"Dataset tabular: {len(parsed)} filas × {len(columns)} columnas", ""]
            for col in columns:
                values = [row[col] for row in parsed if isinstance(row.get(col), (int, float))]
                if len(values) >= 2:
                    lines.append(
                        f"  {col}: min={min(values):.2f}, max={max(values):.2f}, "
                        f"media={statistics.mean(values):.2f}, std={statistics.stdev(values):.2f}"
                    )
                elif values:
                    lines.append(f"  {col}: valor único = {values[0]}")
                else:
                    unique = list({str(row.get(col)) for row in parsed if row.get(col) is not None})
                    lines.append(f"  {col} (categórico): {len(unique)} valores únicos → {unique[:5]}")
            return "\n".join(lines)
    except (json.JSONDecodeError, TypeError, KeyError, statistics.StatisticsError):
        pass

    try:
        reader = csv.DictReader(io.StringIO(data.strip()))
        rows = list(reader)
        if rows:
            columns = list(rows[0].keys())
            lines = [f"Dataset CSV: {len(rows)} filas × {len(columns)} columnas", ""]
            for col in columns:
                values = []
                for row in rows:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, KeyError):
                        pass
                if len(values) >= 2:
                    lines.append(
                        f"  {col}: min={min(values):.2f}, max={max(values):.2f}, "
                        f"media={statistics.mean(values):.2f}, std={statistics.stdev(values):.2f}"
                    )
                elif values:
                    lines.append(f"  {col}: valor único = {values[0]}")
                else:
                    unique = list({row.get(col, "") for row in rows})
                    lines.append(f"  {col} (texto): {len(unique)} valores únicos")
            return "\n".join(lines)
    except Exception:
        pass

    return (
        f"Datos recibidos (formato no estructurado):\n{data}\n\n"
        f"Para un análisis completo, considera:\n"
        f"1. Distribución y estadísticas descriptivas\n"
        f"2. Valores atípicos y datos faltantes\n"
        f"3. Correlaciones entre variables\n"
        f"4. Tendencias temporales (si aplica)\n"
        f"5. Segmentación por categorías clave"
    )


__all__ = ["analyze_data"]
