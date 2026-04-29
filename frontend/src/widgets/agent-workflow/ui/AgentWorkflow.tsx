import { useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { useEffect } from 'react'

type Props = {
  reasoning: string
  conclusion: string
  finalResponse: string
  isThinking: boolean
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

export function AgentWorkflow({ reasoning, conclusion, finalResponse, isThinking, status }: Props) {
  const hasContent = Boolean(reasoning || conclusion || finalResponse)
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
  }, [hasContent, reasoning, conclusion, finalResponse, showThinkingInline])

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
