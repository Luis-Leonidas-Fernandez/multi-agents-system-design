import { useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { useEffect } from 'react'
import type { DashboardArtifact } from '@/shared/types/realtime'

type Props = {
  reasoning: string
  conclusion: string
  finalResponse: string
  artifacts: DashboardArtifact[]
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

export function AgentWorkflow({
  reasoning,
  conclusion,
  finalResponse,
  artifacts,
  googleCalendarEnabled,
  onApproveArtifact,
  onDeleteArtifact,
  onCreateEventsFromArtifact,
  isThinking,
  status,
}: Props) {
  const hasContent = Boolean(reasoning || conclusion || finalResponse || artifacts.length > 0)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({})
  const [copyStatus, setCopyStatus] = useState('')
  const showThinkingInline = isThinking || status === 'thinking' || status === 'responding'

  const transcript = useMemo(
    () => [
      finalResponse ? { id: 'final-response', label: 'Final response', className: 'chat-bubble-final', text: finalResponse } : null,
    ].filter((item): item is { id: string; label: string; className: string; text: string } => item !== null),
    [finalResponse],
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

  const scrollTranscript = (direction: 'up' | 'down' | 'top' | 'bottom') => {
    const element = scrollRef.current
    if (!element) return
    if (direction === 'top') {
      element.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }
    if (direction === 'bottom') {
      element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' })
      return
    }
    element.scrollBy({ top: direction === 'up' ? -180 : 180, behavior: 'smooth' })
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      scrollTranscript('up')
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      scrollTranscript('down')
    }
    if (event.key === 'PageUp') {
      event.preventDefault()
      scrollTranscript('up')
    }
    if (event.key === 'PageDown') {
      event.preventDefault()
      scrollTranscript('down')
    }
    if (event.key === 'Home') {
      event.preventDefault()
      scrollTranscript('top')
    }
    if (event.key === 'End') {
      event.preventDefault()
      scrollTranscript('bottom')
    }
  }

  useEffect(() => {
    const element = scrollRef.current
    if (!element || !hasContent) return
    window.requestAnimationFrame(() => {
      element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' })
    })
  }, [hasContent, reasoning, conclusion, finalResponse, showThinkingInline, artifacts])

  useEffect(() => {
    console.log('[ARTIFACT_DEBUG][frontend] AgentWorkflow render', {
      artifactsCount: artifacts.length,
      artifactPaths: artifacts.map((artifact) => artifact.jsonPath),
    })
  }, [artifacts])

  return (
    <section className="panel workflow-panel chat-panel">
      {status === 'responding' && hasContent ? <div className="workflow-status">Respuesta en camino</div> : null}
      {hasContent ? (
        <>
          <div className="workflow-scroll" ref={scrollRef} tabIndex={0} onKeyDown={handleKeyDown} aria-label="Transcript">
            {transcript.map((entry) => {
              const isExpanded = Boolean(expandedIds[entry.id])
              return (
                <article key={entry.id} className={`chat-bubble ${entry.className} transcript-card`}>
                  <div className="chat-bubble-head">
                    <span className="chat-meta">{entry.label}</span>
                    <div className="chat-bubble-actions">
                      <button type="button" className="bubble-action" onClick={() => copyMessage(entry.id, entry.label, entry.text)}>
                        Copy
                      </button>
                      <button type="button" className="bubble-action" onClick={() => toggleExpanded(entry.id)}>
                        {isExpanded ? 'Collapse' : 'Expand'}
                      </button>
                    </div>
                  </div>
                  <p className={isExpanded ? 'chat-bubble-body' : 'chat-bubble-body is-collapsed'}>{entry.text}</p>
                </article>
              )
            })}

            {artifacts.map((artifact, index) => {
              const rawPreview = expandedIds[`artifact-preview-${artifact.jsonPath}`]
              const selectedPreview = rawPreview === 'json' ? 'json' : 'markdown'
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
                      onClick={() => setExpandedIds((current) => ({ ...current, [`artifact-preview-${artifact.jsonPath}`]: 'json' }))}
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
                      onClick={() => setExpandedIds((current) => ({ ...current, [`artifact-preview-${artifact.jsonPath}`]: 'markdown' }))}
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

          <div className="workflow-nav-row">
            <div className="workflow-nav-buttons" aria-label="Transcript navigation">
              <button type="button" className="bubble-action" onClick={() => scrollTranscript('up')}>↑</button>
              <button type="button" className="bubble-action" onClick={() => scrollTranscript('down')}>↓</button>
              <button type="button" className="bubble-action" onClick={() => scrollTranscript('top')}>Home</button>
              <button type="button" className="bubble-action" onClick={() => scrollTranscript('bottom')}>End</button>
            </div>
            <small className="workflow-hints">Scroll · ↑↓ · PgUp/Dn · Home/End · Copy · Expand</small>
            {copyStatus ? <span className="workflow-copy-status">{copyStatus}</span> : null}
          </div>
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
