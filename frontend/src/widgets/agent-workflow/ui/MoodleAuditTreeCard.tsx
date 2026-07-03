import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import type { MoodleAuditTree, MoodleAuditTreeNode } from '@/shared/types/realtime'

type Props = {
  tree: MoodleAuditTree
}

type ExportableTreeItem = {
  kind: MoodleAuditTreeNode['kind']
  title: string
  url: string
  description: string
  location: string
}

const KIND_ICON: Record<MoodleAuditTreeNode['kind'], string> = {
  course: '◉',
  page: '▣',
  section: '▤',
  forum: '💬',
  quiz: '📝',
  assignment: '📦',
  document: '📄',
  image: '🖼',
  video: '🎬',
  preview: '👁',
  link: '🔗',
  google_slides: '📊',
  google_drive: '▶',
  external_redirect: '↗',
  unknown: '•',
}

const KIND_LABEL: Record<MoodleAuditTreeNode['kind'], string> = {
  course: 'Curso',
  page: 'Página',
  section: 'Sección',
  forum: 'Foro',
  quiz: 'Quiz',
  assignment: 'Entrega',
  document: 'Documento',
  image: 'Imagen',
  video: 'Video',
  preview: 'Preview',
  link: 'Link',
  google_slides: 'Google Slides',
  google_drive: 'Google Drive',
  external_redirect: 'Redirect',
  unknown: 'Recurso',
}

function buildInitialExpanded(node: MoodleAuditTreeNode, depth = 0): Record<string, boolean> {
  const expanded: Record<string, boolean> = { [node.id]: depth <= 1 }
  node.children.forEach((child) => {
    Object.assign(expanded, buildInitialExpanded(child, depth + 1))
  })
  return expanded
}

function metadataEntries(node: MoodleAuditTreeNode) {
  return Object.entries(node.metadata ?? {}).filter(([, value]) => String(value ?? '').trim().length > 0)
}

function statLabel(resourceType: string) {
  return resourceType.replace(/_/g, ' ')
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function kindAccent(kind: MoodleAuditTreeNode['kind']) {
  if (kind === 'course') return 'cyan'
  if (kind === 'section' || kind === 'page') return 'violet'
  if (kind === 'document' || kind === 'google_slides' || kind === 'google_drive') return 'amber'
  if (kind === 'quiz' || kind === 'assignment' || kind === 'forum') return 'rose'
  if (kind === 'video' || kind === 'preview' || kind === 'image') return 'emerald'
  return 'slate'
}

function shortenUrl(value: string) {
  const text = value.trim()
  if (!text) return ''
  if (text.length <= 72) return text
  const keep = Math.max(24, Math.floor(text.length / 4))
  return `${text.slice(0, keep)}...${text.slice(-keep)}`
}

function renderLinkLabel(kind: 'url' | 'preview' | 'download' | 'redirect', href: string) {
  const prefix = {
    url: 'URL',
    preview: 'Preview',
    download: 'Download',
    redirect: 'Redirect',
  }[kind]
  return `${prefix}: ${shortenUrl(href)}`
}

function resourceKindLabel(kind: MoodleAuditTreeNode['kind']) {
  return KIND_LABEL[kind] ?? KIND_LABEL.unknown
}

function normalizeText(value: string) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim()
}

function looksLikeNoiseText(value: string) {
  const normalized = normalizeText(value)
  if (!normalized) return true
  const exactNoise = new Set([
    'sign in',
    'english ‎(en)‎',
    'english (en)',
    'espanol - internacional ‎(es)‎',
    'espanol - internacional (es)',
    'mostrar comentarios',
    'view.php',
    'cancel contracts here',
    'colab paid products',
  ])
  if (exactNoise.has(normalized)) return true
  return (
    normalized.startsWith('sign in') ||
    normalized.includes('service login') ||
    normalized.includes('paid products') ||
    normalized.includes('cancel subscription') ||
    normalized.includes('mostrar comentarios') ||
    normalized === 'english' ||
    normalized === 'espanol - internacional'
  )
}

function optionalNoiseText(value: string | undefined) {
  const trimmed = (value || '').trim()
  if (!trimmed) return false
  return looksLikeNoiseText(trimmed)
}

function isNoiseUrl(value: string) {
  const normalized = normalizeText(value)
  return (
    normalized.includes('accounts.google.com/servicelogin') ||
    normalized.includes('colab.research.google.com/signup') ||
    normalized.includes('colab.research.google.com/cancel-subscription') ||
    normalized.includes('lang=es') ||
    normalized.includes('lang=en') ||
    normalized.includes('&lang=') ||
    normalized.includes('?lang=')
  )
}

function isStudyMaterialNode(node: MoodleAuditTreeNode) {
  const materialKinds = new Set<MoodleAuditTreeNode['kind']>([
    'document',
    'video',
    'google_slides',
    'google_drive',
  ])
  if (!materialKinds.has(node.kind)) return false
  if ((node.badges ?? []).some((badge) => ['submitted', 'can_submit', 'graded', 'locked'].includes(normalizeText(badge)))) {
    return false
  }
  if (looksLikeNoiseText(node.title) || optionalNoiseText(node.subtitle) || optionalNoiseText(node.description)) {
    return false
  }
  return true
}

function preferredExportUrl(node: MoodleAuditTreeNode) {
  if (node.kind === 'document') {
    return node.downloadUrl || node.canonicalUrl || node.redirectUrl || node.url
  }
  if (node.kind === 'google_slides') {
    return node.canonicalUrl || node.redirectUrl || node.url || node.previewUrl
  }
  if (node.kind === 'google_drive') {
    return node.downloadUrl || node.canonicalUrl || node.redirectUrl || node.url || node.previewUrl
  }
  if (node.kind === 'video') {
    return node.url || node.canonicalUrl || node.previewUrl || node.redirectUrl
  }
  return node.redirectUrl || node.url || node.canonicalUrl || node.previewUrl || node.downloadUrl
}

function collectExportableItems(
  node: MoodleAuditTreeNode,
  trail: string[] = [],
  items: ExportableTreeItem[] = [],
) {
  const nextTrail = node.kind === 'course' ? trail : [...trail, node.title]
  const preferredUrl = preferredExportUrl(node)
  if (isStudyMaterialNode(node) && preferredUrl && !isNoiseUrl(preferredUrl)) {
    const location = trail.join(' → ')
    const descriptionParts = [
      resourceKindLabel(node.kind),
      node.subtitle?.trim() || '',
      node.description?.trim() || '',
    ].filter(Boolean)

    items.push({
      kind: node.kind,
      title: node.title,
      url: preferredUrl,
      description: descriptionParts.join(' · '),
      location,
    })
  }

  node.children.forEach((child) => collectExportableItems(child, nextTrail, items))
  return items
}

function buildWordDocumentHtml(tree: MoodleAuditTree) {
  const items = collectExportableItems(tree.root)
  const generatedAt = new Date().toLocaleString('es-AR')
  const sections = items
    .map((item, index) => {
      const kind = escapeHtml(resourceKindLabel(item.kind))
      const title = escapeHtml(item.title)
      const url = escapeHtml(item.url)
      const description = escapeHtml(item.description || kind)
      const location = escapeHtml(item.location || tree.courseName)
      return `
        <div class="resource-card">
          <div class="resource-index">${index + 1}</div>
          <div class="resource-body">
            <h3>${title}</h3>
            <p class="resource-location">${location}</p>
            <p class="resource-url">${url}</p>
            <p class="resource-description">${description}</p>
          </div>
        </div>
      `
    })
    .join('\n')

  return `
    <html xmlns:o="urn:schemas-microsoft-com:office:office"
          xmlns:w="urn:schemas-microsoft-com:office:word"
          xmlns="http://www.w3.org/TR/REC-html40">
      <head>
        <meta charset="utf-8" />
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <title>${escapeHtml(tree.courseName)} - materiales</title>
        <style>
          body { font-family: Calibri, Arial, sans-serif; color: #0f172a; margin: 28px; }
          h1 { font-size: 24px; margin: 0 0 6px; }
          .meta { color: #475569; font-size: 11pt; margin-bottom: 18px; }
          .summary { margin: 0 0 20px; padding: 12px 14px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; }
          .summary p { margin: 4px 0; }
          .resource-card { border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 14px; margin-bottom: 12px; }
          .resource-index { font-size: 10pt; font-weight: 700; color: #0369a1; margin-bottom: 8px; }
          .resource-body h3 { margin: 0 0 4px; font-size: 14pt; }
          .resource-location { margin: 0 0 6px; color: #475569; font-size: 10pt; }
          .resource-url { margin: 0 0 6px; color: #0f766e; font-size: 10pt; word-break: break-all; }
          .resource-description { margin: 0; color: #1e293b; font-size: 10.5pt; }
        </style>
      </head>
      <body>
        <h1>${escapeHtml(tree.courseName)}</h1>
        <p class="meta">Job UID: ${escapeHtml(tree.jobUid)} · Generado: ${escapeHtml(generatedAt)}</p>
        <div class="summary">
          <p><strong>Páginas:</strong> ${tree.stats.pageCount}</p>
          <p><strong>Recursos exportados:</strong> ${items.length}</p>
          <p><strong>Redirects:</strong> ${tree.stats.externalRedirectCount ?? 0}</p>
          <p><strong>Downloads:</strong> ${tree.stats.downloadDocumentCount ?? 0}</p>
        </div>
        ${sections || '<p>No se encontraron materiales exportables en esta auditoría.</p>'}
      </body>
    </html>
  `
}

function formatNodeForClipboard(node: MoodleAuditTreeNode, depth = 0): string[] {
  const indent = '  '.repeat(depth)
  const lines = [`${indent}- [${node.kind}] ${node.title}`]
  if (node.subtitle) lines.push(`${indent}  subtitle: ${node.subtitle}`)
  if (node.description) lines.push(`${indent}  description: ${node.description}`)
  if (node.url) lines.push(`${indent}  url: ${shortenUrl(node.url)}`)
  if (node.canonicalUrl) lines.push(`${indent}  canonical: ${shortenUrl(node.canonicalUrl)}`)
  if (node.previewUrl) lines.push(`${indent}  preview: ${shortenUrl(node.previewUrl)}`)
  if (node.downloadUrl) lines.push(`${indent}  download: ${shortenUrl(node.downloadUrl)}`)
  if (node.redirectUrl) lines.push(`${indent}  redirect: ${shortenUrl(node.redirectUrl)}`)
  if (node.mimeType) lines.push(`${indent}  mime: ${node.mimeType}`)
  if ((node.badges ?? []).length) lines.push(`${indent}  badges: ${node.badges.join(', ')}`)

  const metadata = metadataEntries(node)
  metadata.forEach(([key, value]) => {
    const renderedValue = typeof value === 'string' && /^https?:\/\//i.test(value) ? shortenUrl(value) : String(value)
    lines.push(`${indent}  ${key}: ${renderedValue}`)
  })

  node.children.forEach((child) => {
    lines.push(...formatNodeForClipboard(child, depth + 1))
  })
  return lines
}

function buildClipboardTree(tree: MoodleAuditTree): string {
  const lines = [
    `Materia: ${tree.courseName}`,
    `Job UID: ${tree.jobUid}`,
    `Páginas: ${tree.stats.pageCount}`,
    `Retenidas: ${tree.stats.retainedPageCount ?? tree.stats.pageCount}`,
    `Redirects: ${tree.stats.externalRedirectCount ?? 0}`,
    `Downloads: ${tree.stats.downloadDocumentCount ?? 0}`,
  ]

  const resourceTypeEntries = Object.entries(tree.stats.resourceTypeCounts ?? {}).filter(([, count]) => count > 0)
  if (resourceTypeEntries.length) {
    lines.push('Tipos de recurso:')
    resourceTypeEntries.forEach(([resourceType, count]) => {
      lines.push(`- ${resourceType}: ${count}`)
    })
  }

  lines.push('', 'Árbol:')
  lines.push(...formatNodeForClipboard(tree.root))
  return lines.join('\n')
}

export function MoodleAuditTreeCard({ tree }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [copyStatus, setCopyStatus] = useState('')
  const [downloadStatus, setDownloadStatus] = useState('')
  const initialExpanded = useMemo(() => buildInitialExpanded(tree.root), [tree])

  useEffect(() => {
    setExpanded(initialExpanded)
  }, [initialExpanded, tree.jobUid])

  const toggleNode = (nodeId: string) => {
    setExpanded((current) => ({ ...current, [nodeId]: !current[nodeId] }))
  }

  const copyTree = async () => {
    const text = buildClipboardTree(tree)
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = text
        textarea.setAttribute('readonly', 'true')
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
      }
      setCopyStatus('Árbol copiado')
      window.setTimeout(() => setCopyStatus(''), 1600)
    } catch {
      setCopyStatus('No se pudo copiar el árbol')
      window.setTimeout(() => setCopyStatus(''), 1600)
    }
  }

  const downloadDocs = () => {
    try {
      const html = buildWordDocumentHtml(tree)
      const blob = new Blob(['\ufeff', html], { type: 'application/msword;charset=utf-8' })
      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      const safeCourseName = tree.courseName
        .normalize('NFKD')
        .replace(/[^\w\s-]/g, '')
        .trim()
        .replace(/\s+/g, '-')
        .toLowerCase() || 'materia'
      anchor.href = objectUrl
      anchor.download = `${safeCourseName}-materiales.doc`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      URL.revokeObjectURL(objectUrl)
      setDownloadStatus('Documento Word descargado')
      window.setTimeout(() => setDownloadStatus(''), 1800)
    } catch {
      setDownloadStatus('No se pudo generar el Word')
      window.setTimeout(() => setDownloadStatus(''), 1800)
    }
  }

  const renderNode = (node: MoodleAuditTreeNode, depth = 0) => {
    const hasChildren = node.children.length > 0
    const isExpanded = expanded[node.id] ?? depth <= 1
    const metadata = metadataEntries(node)
    const tone = kindAccent(node.kind)
    const isRoot = depth === 0

    return (
      <li key={node.id} className={`moodle-tree-node moodle-tree-node-${node.kind}`}>
        <div className={`moodle-tree-row ${isRoot ? 'moodle-tree-row-root' : ''}`} style={{ '--tree-depth': depth } as CSSProperties}>
          <div className="moodle-tree-row-main">
            {hasChildren ? (
              <button
                type="button"
                className="moodle-tree-toggle"
                onClick={() => toggleNode(node.id)}
                aria-label={isExpanded ? 'Colapsar rama' : 'Expandir rama'}
              >
                {isExpanded ? '−' : '+'}
              </button>
            ) : (
              <span className="moodle-tree-toggle moodle-tree-toggle-static" aria-hidden="true">
                ·
              </span>
            )}
            <span className={`moodle-tree-icon moodle-tree-icon-${node.kind}`} aria-hidden="true">
              {KIND_ICON[node.kind] ?? KIND_ICON.unknown}
            </span>
            <div className={`moodle-tree-node-card moodle-tree-node-card-${tone} ${isRoot ? 'moodle-tree-node-card-root' : ''}`}>
              <div className="moodle-tree-copy">
                <div className="moodle-tree-title-row">
                  <strong>{node.title}</strong>
                  <span className={`moodle-tree-kind-pill moodle-tree-kind-pill-${tone}`}>{KIND_LABEL[node.kind] ?? KIND_LABEL.unknown}</span>
                  {(node.badges ?? []).slice(0, 5).map((badge) => (
                    <span key={`${node.id}-${badge}`} className="moodle-tree-badge">
                      {badge}
                    </span>
                  ))}
                </div>
                {node.subtitle ? <div className="moodle-tree-subtitle">{node.subtitle}</div> : null}
                {node.description ? <p className="moodle-tree-description">{node.description}</p> : null}
                {node.url || node.previewUrl || node.downloadUrl || node.redirectUrl ? (
                  <div className="moodle-tree-links">
                    {node.url ? (
                      <a href={node.url} target="_blank" rel="noreferrer" className="moodle-tree-link-pill">
                        {renderLinkLabel('url', node.url)}
                      </a>
                    ) : null}
                    {node.previewUrl ? (
                      <a href={node.previewUrl} target="_blank" rel="noreferrer" className="moodle-tree-link-pill">
                        {renderLinkLabel('preview', node.previewUrl)}
                      </a>
                    ) : null}
                    {node.downloadUrl ? (
                      <a href={node.downloadUrl} target="_blank" rel="noreferrer" className="moodle-tree-link-pill">
                        {renderLinkLabel('download', node.downloadUrl)}
                      </a>
                    ) : null}
                    {node.redirectUrl ? (
                      <a href={node.redirectUrl} target="_blank" rel="noreferrer" className="moodle-tree-link-pill">
                        {renderLinkLabel('redirect', node.redirectUrl)}
                      </a>
                    ) : null}
                  </div>
                ) : null}
                {metadata.length ? (
                  <div className="moodle-tree-metadata">
                    {metadata.slice(0, 6).map(([key, value]) => (
                      <span key={`${node.id}-${key}`} className="moodle-tree-meta-pill">
                        <b>{key}</b>: {String(value)}
                      </span>
                    ))}
                    {metadata.length > 6 ? <span className="moodle-tree-meta-pill">+{metadata.length - 6} más</span> : null}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
        {hasChildren && isExpanded ? <ul className="moodle-tree-children">{node.children.map((child) => renderNode(child, depth + 1))}</ul> : null}
      </li>
    )
  }

  const resourceTypeEntries = Object.entries(tree.stats.resourceTypeCounts ?? {}).filter(([, count]) => count > 0)

  return (
    <article className="chat-bubble chat-bubble-assistant transcript-card moodle-tree-card">
      <div className="chat-bubble-head moodle-tree-card-head">
        <div>
          <span className="chat-meta">Moodle audit tree</span>
          <div className="moodle-tree-card-title">
            <strong>{tree.courseName}</strong>
            <span className="moodle-tree-job">{tree.jobUid}</span>
          </div>
        </div>
      </div>

      {copyStatus ? <div className="workflow-copy-status">{copyStatus}</div> : null}
      {downloadStatus ? <div className="workflow-copy-status">{downloadStatus}</div> : null}

      <div className="moodle-tree-stats">
        <span>Páginas: <strong>{tree.stats.pageCount}</strong></span>
        <span>Retenidas: <strong>{tree.stats.retainedPageCount ?? tree.stats.pageCount}</strong></span>
        <span>Redirects: <strong>{tree.stats.externalRedirectCount ?? 0}</strong></span>
        <span>Downloads: <strong>{tree.stats.downloadDocumentCount ?? 0}</strong></span>
      </div>

      {resourceTypeEntries.length ? (
        <div className="moodle-tree-resource-pills">
          {resourceTypeEntries.map(([resourceType, count]) => (
            <span key={resourceType} className="moodle-tree-resource-pill">
              {count} {statLabel(resourceType)}
            </span>
          ))}
        </div>
      ) : null}
      <div className="moodle-tree-shell">
        <ul className="moodle-tree-root">{renderNode(tree.root)}</ul>
      </div>
      <div className="moodle-tree-footer-actions">
        <button type="button" className="bubble-action bubble-action-primary" onClick={downloadDocs}>
          Descargar docs
        </button>
        <button type="button" className="bubble-action" onClick={copyTree}>
          Copiar árbol
        </button>
      </div>
    </article>
  )
}
