import { useEffect, useState } from 'react'
import type { KeyboardEvent } from 'react'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAbort: () => void
  googleCalendarEnabled: boolean
  onToggleGoogleCalendar: () => void
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

export function SendAgentAction({
  value,
  onChange,
  onSend,
  onAbort,
  googleCalendarEnabled,
  onToggleGoogleCalendar,
  status,
}: Props) {
  const isLocked = status === 'thinking' || status === 'responding'
  const [isSubmitting, setIsSubmitting] = useState(false)
  const buttonLabel = isLocked ? 'Abortar' : 'Send'

  useEffect(() => {
    if (isLocked) {
      setIsSubmitting(false)
      return
    }
    if (status === 'idle' || status === 'error') {
      setIsSubmitting(false)
    }
  }, [isLocked, status])

  const handleSend = () => {
    if (!value.trim() || isLocked) return
    setIsSubmitting(true)
    onSend()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') return
    event.preventDefault()
    handleSend()
  }

  return (
    <div className="composer composer-stateful">
      <button
        type="button"
        className={`mcp-toggle ${googleCalendarEnabled ? 'mcp-toggle-enabled' : ''}`}
        onClick={onToggleGoogleCalendar}
        aria-pressed={googleCalendarEnabled}
        title={googleCalendarEnabled ? 'Google Calendar MCP activado' : 'Google Calendar MCP desactivado'}
      >
        <span className="mcp-toggle-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" className="mcp-toggle-icon-svg">
            <path d="M6.4 3.8H17.6C19.37 3.8 20.8 5.23 20.8 7V17.1C20.8 17.75 20.54 18.37 20.09 18.83L18.78 20.14C18.33 20.59 17.72 20.85 17.08 20.85H6.4C4.63 20.85 3.2 19.42 3.2 17.65V7C3.2 5.23 4.63 3.8 6.4 3.8Z" fill="#fff"/>
            <path d="M6.4 3.8H17.6C19.37 3.8 20.8 5.23 20.8 7V8.25H3.2V7C3.2 5.23 4.63 3.8 6.4 3.8Z" fill="#4285F4"/>
            <path d="M20.8 8.25V17.1C20.8 17.75 20.54 18.37 20.09 18.83L18.78 20.14C18.33 20.59 17.72 20.85 17.08 20.85H16.95V8.25H20.8Z" fill="#FBBC05"/>
            <path d="M3.2 8.25H7.1V20.85H6.4C4.63 20.85 3.2 19.42 3.2 17.65V8.25Z" fill="#34A853"/>
            <path d="M7.1 16.95H18.92L18.78 20.14C18.33 20.59 17.72 20.85 17.08 20.85H6.4C5.12 20.85 4.01 20.09 3.5 19L7.1 16.95Z" fill="#34A853"/>
            <path d="M16.95 16.95H20.33C20.18 17.66 19.82 18.31 19.29 18.83L18.78 19.34L16.95 16.95Z" fill="#EA4335"/>
            <path d="M7.8 11.1H16.2V16.4H7.8V11.1Z" fill="#fff"/>
            <path d="M10.1 15.35V14.4C10.52 14.7 11.02 14.88 11.57 14.88C12.36 14.88 12.91 14.47 12.91 13.84C12.91 13.22 12.41 12.88 11.72 12.88H11.08V12.05H11.63C12.29 12.05 12.76 11.72 12.76 11.16C12.76 10.63 12.33 10.28 11.71 10.28C11.17 10.28 10.68 10.49 10.29 10.83V9.92C10.69 9.65 11.23 9.48 11.8 9.48C13.11 9.48 13.94 10.16 13.94 11.07C13.94 11.8 13.51 12.33 12.79 12.58V12.61C13.61 12.77 14.11 13.3 14.11 14.12C14.11 15.27 13.08 16 11.72 16C11.08 16 10.5 15.79 10.1 15.35Z" fill="#4285F4"/>
          </svg>
        </span>
      </button>
      <label className="sr-only" htmlFor="agent-message">Composer</label>
      <input
        id="agent-message"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Preguntá sobre tus materias"
        aria-label="Preguntá sobre tus materias"
        onKeyDown={handleKeyDown}
      />
      <button
        className="send-button"
        onClick={isLocked ? onAbort : handleSend}
        aria-label={buttonLabel}
        title={buttonLabel}
      >
        {isSubmitting ? (
          <span className="send-button-spinner" aria-hidden="true" />
        ) : isLocked ? (
          <span className="send-button-stop" aria-hidden="true" />
        ) : (
          <span className="send-button-icon" aria-hidden="true">→</span>
        )}
      </button>
    </div>
  )
}
