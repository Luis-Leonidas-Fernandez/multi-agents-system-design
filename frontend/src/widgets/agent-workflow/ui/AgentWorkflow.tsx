import { useMemo, useRef, useState } from 'react'
import { useEffect } from 'react'
import type { DashboardArtifact, MoodleAuditTree } from '@/shared/types/realtime'
import { MoodleAuditTreeCard } from '@/widgets/agent-workflow/ui/MoodleAuditTreeCard'

type Props = {
  lastUserMessage: string
  reasoning: string
  conclusion: string
  finalResponse: string
  artifacts: DashboardArtifact[]
  moodleAuditTree: MoodleAuditTree | null
  googleCalendarEnabled: boolean
  onApproveArtifact: (artifactPath: string) => void
  onDeleteArtifact: (artifactPath: string) => void
  onCreateEventsFromArtifact: (artifactPath: string) => void
  isThinking: boolean
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

function fileNameFromPath(path: string) {
  return path.split('/').filter(Boolean).pop() || path
}

function normalizeMarkdownSpacing(text: string) {
  return text
    .replace(/\s+(#{3,6}\s+)/g, '\n\n$1')
    .replace(/\s+(- \*\*)/g, '\n$1')
    .replace(/\s+(> )/g, '\n$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function sanitizeAuditResponse(text: string) {
  if (!text.trim()) return text
  return normalizeMarkdownSpacing(text)
    .split('\n')
    .filter((line) => {
      const normalized = line.trim().toLowerCase()
      return !(
        normalized.startsWith('- json audit:') ||
        normalized.startsWith('- schema:') ||
        normalized.startsWith('- resumen:')
      )
    })
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}


function formatMessageForCopy(text: string) {
  return normalizeMarkdownSpacing(text)
    .split('\n')
    .map((rawLine) => {
      const line = rawLine.trim()
      if (!line) return ''
      return line
        .replace(/^#{3,6}\s+/, '')
        .replace(/^>\s?/, '')
        .replace(/^-\s+/, '• ')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
    })
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

function renderFormattedMessage(text: string) {
  return text.split('\n').map((rawLine, index) => {
    const line = rawLine.trim()
    const key = `${index}-${line.slice(0, 24)}`
    if (!line) return <span key={key} className="formatted-response-spacer" aria-hidden="true" />

    const heading = /^(#{3,6})\s+(.*)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const className = `formatted-response-heading formatted-response-heading-${level}`
      return <div key={key} className={className}>{renderInlineMarkdown(heading[2])}</div>
    }

    const bullet = /^-\s+(.*)$/.exec(line)
    if (bullet) {
      return <div key={key} className="formatted-response-line formatted-response-bullet"><span aria-hidden="true">•</span><span>{renderInlineMarkdown(bullet[1])}</span></div>
    }

    const quote = /^>\s?(.*)$/.exec(line)
    if (quote) {
      return <blockquote key={key} className="formatted-response-quote">{renderInlineMarkdown(quote[1])}</blockquote>
    }

    return <div key={key} className="formatted-response-line">{renderInlineMarkdown(line)}</div>
  })
}

export function AgentWorkflow({
  lastUserMessage,
  reasoning,
  conclusion,
  finalResponse,
  artifacts,
  moodleAuditTree,
  googleCalendarEnabled,
  onApproveArtifact,
  onDeleteArtifact,
  onCreateEventsFromArtifact,
  isThinking,
  status,
}: Props) {
  const hasContent = Boolean(lastUserMessage || reasoning || conclusion || finalResponse || artifacts.length > 0 || moodleAuditTree)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({})
  const [artifactPreviewMode, setArtifactPreviewMode] = useState<Record<string, 'json' | 'markdown'>>({})
  const [copyStatus, setCopyStatus] = useState('')
  const showThinkingInline = isThinking || status === 'thinking' || status === 'responding'
  const cleanedFinalResponse = useMemo(() => sanitizeAuditResponse(finalResponse), [finalResponse])

  const transcript = useMemo(
    () => [
      lastUserMessage ? { id: 'last-user-message', label: 'Tu pregunta', className: 'chat-bubble-user', text: lastUserMessage, kind: 'user' as const } : null,
      cleanedFinalResponse ? { id: 'final-response', label: 'Respuesta', className: 'chat-bubble-final', text: cleanedFinalResponse, kind: 'assistant' as const } : null,
    ].filter((item): item is { id: string; label: string; className: string; text: string; kind: 'user' | 'assistant' } => item !== null),
    [cleanedFinalResponse, lastUserMessage],
  )

  const toggleExpanded = (id: string) => {
    setExpandedIds((current) => ({ ...current, [id]: !current[id] }))
  }

  const copyMessage = async (id: string, label: string, text: string) => {
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
      setCopyStatus(`${label} copiado`)
      window.setTimeout(() => setCopyStatus(''), 1600)
    } catch {
      setCopyStatus(`No se pudo copiar ${id}`)
      window.setTimeout(() => setCopyStatus(''), 1600)
    }
  }

  useEffect(() => {
    const element = scrollRef.current
    if (!element || !hasContent) return
    window.requestAnimationFrame(() => {
      element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' })
    })
  }, [hasContent, lastUserMessage, reasoning, conclusion, finalResponse, showThinkingInline, artifacts, moodleAuditTree])

  useEffect(() => {
    console.log('[ARTIFACT_DEBUG][frontend] AgentWorkflow render', {
      artifactsCount: artifacts.length,
      artifactPaths: artifacts.map((artifact) => artifact.jsonPath),
    })
  }, [artifacts])

  return (
    <section className={`panel workflow-panel chat-panel ${hasContent ? 'chat-panel-has-content' : 'chat-panel-is-empty'}`}>
      {status === 'responding' && hasContent ? <div className="workflow-status">Respuesta en camino</div> : null}
      {hasContent ? (
        <>
          <div className="workflow-scroll" ref={scrollRef} aria-label="Transcript">
            {transcript.map((entry) => {
              const isExpanded = entry.kind === 'assistant' ? expandedIds[entry.id] ?? true : Boolean(expandedIds[entry.id])
              return (
                <article key={entry.id} className={`chat-bubble ${entry.className} transcript-card`}>
                  <div className="chat-bubble-head">
                    <span className="chat-meta chat-meta-with-icon">
                      {entry.kind === 'user' ? (
                        <span className="chat-meta-icon" aria-hidden="true">
                          <svg viewBox="0 0 24 24" focusable="false">
                            <path d="M12 4C7.03 4 3 7.42 3 11.64c0 2.16 1.06 4.1 2.77 5.46L5.2 20l3.38-1.67c1.05.38 2.2.59 3.42.59 4.97 0 9-3.42 9-7.64C21 7.42 16.97 4 12 4Z" fill="currentColor" />
                          </svg>
                        </span>
                      ) : null}
                      {entry.label}
                    </span>
                    <div className="chat-bubble-actions">
                      <button type="button" className="bubble-action" onClick={() => copyMessage(entry.id, entry.label, entry.kind === 'assistant' ? formatMessageForCopy(entry.text) : entry.text)}>
                        Copy
                      </button>
                      <button type="button" className="bubble-action" onClick={() => toggleExpanded(entry.id)}>
                        {isExpanded ? 'Collapse' : 'Expand'}
                      </button>
                    </div>
                  </div>
                  <div className={isExpanded ? 'chat-bubble-body formatted-response' : 'chat-bubble-body formatted-response is-collapsed'}>{renderFormattedMessage(entry.text)}</div>
                  {entry.kind === 'user' ? (
                    <div className="chat-bubble-pending">
                      <span className="chat-bubble-sent-dot" aria-hidden="true" />
                      <span>Enviado</span>
                    </div>
                  ) : null}
                </article>
              )
            })}

            {artifacts.map((artifact, index) => {
              const selectedPreview = artifactPreviewMode[artifact.jsonPath] ?? 'markdown'
              const previewLabel = selectedPreview === 'json' ? 'JSON' : 'Markdown'
              const previewContent = selectedPreview === 'json' ? artifact.jsonContent : artifact.markdownContent
              return (
                <article key={artifact.jsonPath} className="chat-bubble chat-bubble-assistant transcript-card artifact-card">
                  <div className="chat-bubble-head artifact-head">
                    <div>
                      <span className="chat-meta">Archivos generados #{artifacts.length - index}</span>
                      <div className="artifact-badges">
                        <span className={`artifact-badge ${artifact.structureValid ? 'artifact-badge-valid' : 'artifact-badge-invalid'}`}>
                          {artifact.structureValid ? 'Estructura válida' : 'Estructura inválida'}
                        </span>
                        {artifact.approved ? <span className="artifact-badge artifact-badge-approved">Aprobado</span> : null}
                        {artifact.syncCreatedCount > 0 ? <span className="artifact-badge artifact-badge-synced">Eventos creados: {artifact.syncCreatedCount}</span> : null}
                      </div>
                    </div>
                    <div className="chat-bubble-actions artifact-actions">
                      {!artifact.approved && artifact.structureValid ? (
                        <button type="button" className="bubble-action bubble-action-primary" onClick={() => onApproveArtifact(artifact.jsonPath)}>
                          Aprobar
                        </button>
                      ) : null}
                      {!artifact.structureValid ? (
                        <button type="button" className="bubble-action bubble-action-danger" onClick={() => onDeleteArtifact(artifact.jsonPath)} title="Eliminar JSON inválido">
                          🗑
                        </button>
                      ) : null}
                      {artifact.approved ? (
                        <button
                          type="button"
                          className="bubble-action bubble-action-primary"
                          onClick={() => onCreateEventsFromArtifact(artifact.jsonPath)}
                          disabled={!googleCalendarEnabled}
                          title={googleCalendarEnabled ? 'Crear eventos con este JSON' : 'Activá Calendar para crear eventos'}
                        >
                          Crear eventos
                        </button>
                      ) : null}
                    </div>
                  </div>

                  <div className="artifact-summary-grid">
                    <span>Válidas: <strong>{artifact.validCount}</strong></span>
                    <span>Inválidas: <strong>{artifact.invalidCount}</strong></span>
                    <span>Issues: <strong>{artifact.issueCount}</strong></span>
                  </div>

                  <div className="artifact-files">
                    <button
                      type="button"
                      className={`artifact-file ${selectedPreview === 'json' ? 'artifact-file-active' : ''}`}
                      onClick={() => setArtifactPreviewMode((current) => ({ ...current, [artifact.jsonPath]: 'json' }))}
                    >
                      <span className="artifact-file-icon">{'{ }'}</span>
                      <span className="artifact-file-copy">
                        <strong>{fileNameFromPath(artifact.jsonPath)}</strong>
                        <small>JSON · abrir preview</small>
                      </span>
                    </button>
                    <button
                      type="button"
                      className={`artifact-file ${selectedPreview === 'markdown' ? 'artifact-file-active' : ''}`}
                      onClick={() => setArtifactPreviewMode((current) => ({ ...current, [artifact.jsonPath]: 'markdown' }))}
                    >
                      <span className="artifact-file-icon">MD</span>
                      <span className="artifact-file-copy">
                        <strong>{fileNameFromPath(artifact.markdownPath)}</strong>
                        <small>Markdown · abrir preview</small>
                      </span>
                    </button>
                  </div>

                  {artifact.issues.length ? (
                    <ul className="artifact-issues">
                      {artifact.issues.map((issue, issueIndex) => (
                        <li key={`${artifact.jsonPath}-issue-${issueIndex}`}>
                          <strong>{issue.severity || 'issue'}</strong> · {issue.assignment_title || 'sin título'} · {issue.message || ''}
                        </li>
                      ))}
                    </ul>
                  ) : null}

                  <div className="artifact-preview-head">
                    <span className="artifact-preview-label">Preview {previewLabel}</span>
                  </div>
                  <pre className="artifact-viewer artifact-preview">{previewContent}</pre>
                </article>
              )
            })}

            {moodleAuditTree ? <MoodleAuditTreeCard tree={moodleAuditTree} /> : null}

            {showThinkingInline ? (
              <article className="chat-bubble chat-bubble-assistant transcript-card workflow-thinking-inline">
                <div className="chat-bubble-head">
                  <span className="chat-meta">Pensando</span>
                  <span className="thinking-dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                </div>
                <p className="chat-bubble-body">Buscando y armando la próxima respuesta debajo del último mensaje.</p>
              </article>
            ) : null}
          </div>
          {copyStatus ? <span className="workflow-copy-status">{copyStatus}</span> : null}
        </>
      ) : isThinking ? (
        <div className="workflow-empty workflow-thinking" aria-live="polite" aria-label="Pensando">
          <span className="thinking-label">Pensando</span>
          <span className="thinking-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        </div>
      ) : status === 'error' ? (
        <div className="workflow-empty workflow-error" aria-live="polite">
          <span className="thinking-label">Sin conexión</span>
        </div>
      ) : (
        <div className="workflow-empty" aria-hidden="true" />
      )}
    </section>
  )
}
