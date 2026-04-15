"""Resolución de paths de secciones de prensa local por dominio y tópico."""
from __future__ import annotations

from urllib.parse import urljoin

from domain.topic_detector import detect_news_topic


# Paths curados por dominio → tópico → lista de (path, label).
# Sólo existen para dominios con estructura de secciones conocida.
# Para el resto se usa GENERIC_SECTION_PATHS como fallback.
COUNTRY_PRESS_SECTION_PATHS: dict[str, dict[str, list[tuple[str, str]]]] = {
    # ── ITALIA ─────────────────────────────────────────────────────────────
    "ansa.it": {
        "security": [("/sito/notizie/cronaca/cronaca.shtml", "cronaca")],
        "politics": [("/sito/notizie/politica/politica.shtml", "politica")],
        "economy": [("/sito/notizie/economia/economia.shtml", "economia")],
        "default": [("/sito/notizie/cronaca/cronaca.shtml", "cronaca")],
    },
    "repubblica.it": {
        "security": [("/cronaca/", "cronaca")],
        "politics": [("/politica/", "politica")],
        "default": [("/cronaca/", "cronaca")],
    },
    "ilmessaggero.it": {
        "security": [("/italia/", "italia"), ("/roma/", "roma")],
        "politics": [("/politica/", "politica"), ("/italia/", "italia")],
        "default": [("/italia/", "italia")],
    },
    "ilfattoquotidiano.it": {
        "security": [("/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "ilfoglio.it": {
        "security": [("/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "ilmanifesto.it": {
        "security": [("/", "homepage-cronaca")],
        "politics": [("/sezioni/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "huffingtonpost.it": {
        "security": [("/news/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    # ── ESPAÑA ─────────────────────────────────────────────────────────────
    "elpais.com": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/espana/", "españa"), ("/actualidad/", "actualidad")],
    },
    "elmundo.es": {
        "security": [("/espana/", "españa"), ("/cronica/", "crónica")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/mercados/", "mercados")],
        "default": [("/espana/", "españa")],
    },
    "abc.es": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "lavanguardia.com": {
        "security": [("/sucesos/", "sucesos"), ("/vida/sucesos-y-tribunales/", "sucesos")],
        "politics": [("/politica/", "política"), ("/internacional/", "internacional")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/politica/", "política"), ("/vida/", "vida")],
    },
    "elconfidencial.com": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/mercados/", "mercados")],
        "default": [("/espana/", "españa")],
    },
    "20minutos.es": {
        "security": [("/nacional/", "nacional"), ("/sociedad/", "sociedad")],
        "politics": [("/politica/", "política"), ("/nacional/", "nacional")],
        "economy": [("/economia/", "economía")],
        "default": [("/nacional/", "nacional")],
    },
    "eldiario.es": {
        "security": [("/sociedad/", "sociedad"), ("/espana/", "españa")],
        "politics": [("/politica/", "política"), ("/espana/", "españa")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "publico.es": {
        "security": [("/sociedad/", "sociedad"), ("/espana/", "españa")],
        "politics": [("/politica/", "política"), ("/espana/", "españa")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "larazon.es": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "default": [("/espana/", "españa")],
    },
    "cadenaser.com": {
        "security": [("/noticias/nacional/", "nacional"), ("/noticias/sociedad/", "sociedad")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/nacional/", "nacional")],
        "default": [("/noticias/", "noticias")],
    },
    "rtve.es": {
        "security": [("/noticias/espana/", "españa"), ("/noticias/sociedad/", "sociedad")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/espana/", "españa")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/", "noticias")],
    },
    # ── ARGENTINA ──────────────────────────────────────────────────────────
    "lanacion.com.ar": {
        "security": [("/seguridad/", "seguridad"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/el-mundo/", "el-mundo")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/ultimo-momento/", "último momento"), ("/", "homepage")],
    },
    "infobae.com": {
        "security": [("/sociedad/", "sociedad"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/america/america-latina/", "america")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/sociedad/", "sociedad"), ("/", "homepage")],
    },
    "pagina12.com.ar": {
        "security": [("/secciones/el-pais/", "el-país"), ("/secciones/sociedad/", "sociedad")],
        "politics": [("/secciones/el-pais/", "el-país"), ("/secciones/", "secciones")],
        "economy": [("/secciones/economia/", "economía"), ("/secciones/", "secciones")],
        "default": [("/secciones/el-pais/", "el-país")],
    },
    "perfil.com": {
        "security": [("/noticias/policial/", "policial"), ("/noticias/policial.html", "policial")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/politica.html", "política")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/", "noticias")],
    },
    "cronica.com.ar": {
        "security": [("/categoria/policiales/", "policiales"), ("/categoria/", "noticias")],
        "politics": [("/categoria/politica/", "política")],
        "economy": [("/categoria/economia/", "economía")],
        "default": [("/categoria/policiales/", "policiales")],
    },
    "clarin.com": {
        "security": [("/policiales/", "policiales"), ("/sociedad/", "sociedad")],
        "politics": [("/politica/", "política"), ("/zona/", "zona")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/ultimo-momento/", "último momento")],
    },
    "ambito.com": {
        "security": [("/politica/", "política"), ("/economia/", "economía")],
        "politics": [("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/politica/", "política")],
    },
    "tiempoar.com.ar": {
        "security": [("/secciones/el-pais/", "el-país"), ("/", "homepage")],
        "politics": [("/secciones/el-pais/", "el-país")],
        "default": [("/", "homepage")],
    },
    # ── CHILE ──────────────────────────────────────────────────────────────
    "emol.com": {
        "security": [("/noticias/nacional/", "nacional"), ("/noticias/policial/", "policial")],
        "politics": [("/noticias/nacional/", "nacional"), ("/noticias/politica/", "política")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/nacional/", "nacional")],
    },
    "latercera.com": {
        "security": [("/nacional/", "nacional"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/nacional/", "nacional")],
        "economy": [("/pulso/", "pulso"), ("/negocios/", "negocios")],
        "default": [("/nacional/", "nacional")],
    },
    # ── MÉXICO ─────────────────────────────────────────────────────────────
    "eluniversal.com.mx": {
        "security": [("/nacion/seguridad/", "seguridad"), ("/estados/", "estados")],
        "politics": [("/nacion/politica/", "política"), ("/nacion/", "nación")],
        "economy": [("/finanzas/", "finanzas"), ("/economia/", "economía")],
        "default": [("/nacion/", "nación")],
    },
    "milenio.com": {
        "security": [("/policia/", "policía"), ("/estados/", "estados")],
        "politics": [("/politica/", "política"), ("/mexico/", "méxico")],
        "economy": [("/negocios/", "negocios")],
        "default": [("/policia/", "policía")],
    },
    # ── COLOMBIA ───────────────────────────────────────────────────────────
    "eltiempo.com": {
        "security": [("/justicia/", "justicia"), ("/colombia/", "colombia")],
        "politics": [("/politica/", "política"), ("/colombia/", "colombia")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/colombia/", "colombia")],
    },
}

# Paths genéricos usados cuando el dominio no tiene entrada curada.
GENERIC_SECTION_PATHS: dict[str, list[tuple[str, str]]] = {
    "security": [
        ("/seguridad/", "seguridad"),
        ("/policiales/", "policiales"),
        ("/sociedad/", "sociedad"),
        ("/sucesos/", "sucesos"),
        ("/espana/", "españa"),
        ("/nacional/", "nacional"),
        ("/cronaca/", "cronaca"),
        ("/", "homepage"),
    ],
    "politics": [
        ("/politica/", "política"),
        ("/espana/", "españa"),
        ("/nacional/", "nacional"),
        ("/gobierno/", "gobierno"),
        ("/nacion/", "nación"),
        ("/secciones/el-pais/", "el-país"),
        ("/", "homepage"),
    ],
    "economy": [
        ("/economia/", "economía"),
        ("/finanzas/", "finanzas"),
        ("/negocios/", "negocios"),
        ("/mercados/", "mercados"),
        ("/", "homepage"),
    ],
    "default": [
        ("/noticias/", "noticias"),
        ("/actualidad/", "actualidad"),
        ("/ultimo-momento/", "último momento"),
        ("/nacional/", "nacional"),
        ("/espana/", "españa"),
        ("/", "homepage"),
    ],
}


def build_country_press_section_targets(
    domain: str,
    fallback_url: str,
    last_message: str,
) -> list[tuple[str, str]]:
    """Construye la lista de URLs de secciones a scrapear para un dominio dado.

    Usa paths curados si existen; cae a genéricos si no.
    Retorna máximo 4 URLs distintas.
    """
    topic = detect_news_topic(last_message)
    base = (fallback_url or f"https://{domain}/").strip() or f"https://{domain}/"
    if not base.endswith("/"):
        base = base + "/"

    domain_map = COUNTRY_PRESS_SECTION_PATHS.get(domain, {})
    candidates = list(domain_map.get(topic) or domain_map.get("default") or [])
    if not candidates:
        candidates = list(GENERIC_SECTION_PATHS.get(topic, GENERIC_SECTION_PATHS["default"]))

    built: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, label in candidates:
        full_url = base if path == "/" else urljoin(base, path.lstrip("/"))
        if full_url in seen:
            continue
        seen.add(full_url)
        built.append((full_url, label))

    return built[:4]
