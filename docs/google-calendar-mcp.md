# Google Calendar con Codex

Este repo incluye un servidor MCP local para conectar Codex con Google Calendar.

## Qué hace

- Lista eventos
- Crea eventos
- Actualiza eventos
- Borra eventos

## Archivo del servidor

`/Users/luis/Desktop/PROYECTOS/06_multi_agents/integrations/google_calendar_mcp.py`

## Variables de entorno requeridas

Configurá estas variables antes de arrancar el servidor:

- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REFRESH_TOKEN`

Opcional:

- `GOOGLE_CALENDAR_ID` — por defecto usa `primary`

También podés guardar estas variables en:

- `~/.codex/google-calendar.env` para usarlo desde cualquier proyecto
- `./.env` en este repo, si preferís probar localmente

## Scope de OAuth

Usá este scope para eventos de calendario:

`https://www.googleapis.com/auth/calendar.events`

## Configuración global de Codex

Agregá este MCP server a `~/.codex/config.toml`:

```toml
[mcp_servers.google_calendar]
command = "/usr/bin/python3"
args = ["/Users/luis/Desktop/PROYECTOS/06_multi_agents/integrations/google_calendar_mcp.py"]
```

## Nota importante

El servidor necesita un `refresh_token` válido de Google OAuth para renovar el access token automáticamente.
Si no tenés esas credenciales, primero hay que generarlas en Google Cloud o con OAuth Playground.
