# Comandos del proyecto

## Levantar el proyecto

```bash
python main.py
```

## Levantar bridge para frontend

```bash
python main.py --frontend-bridge
```

## Modo desarrollo

```bash
make dev
```

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest tests/ -v
```

## Tests puntuales

```bash
pytest tests/test_web_scraping_node.py -v
pytest tests/test_modules_smoke.py -v
pytest tests/test_web_source_policy.py -v
```
