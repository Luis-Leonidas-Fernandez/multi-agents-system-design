import type { KeyboardEvent } from 'react'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAbort: () => void
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

export function SendAgentAction({ value, onChange, onSend, onAbort, status }: Props) {
  const isLocked = status === 'thinking' || status === 'responding'
  const buttonContent = isLocked ? '⏹' : 'Send'
  const buttonLabel = isLocked ? 'Abortar' : 'Send'

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') return
    event.preventDefault()
    if (!value.trim() || isLocked) return
    onSend()
  }

  return (
    <div className="composer composer-stateful">
      <label className="sr-only" htmlFor="agent-message">Composer</label>
      <input
        id="agent-message"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Ask the agent..."
        aria-label="Ask the agent"
        onKeyDown={handleKeyDown}
      />
      <button
        className="send-button"
        onClick={isLocked ? onAbort : onSend}
        aria-label={buttonLabel}
        title={buttonLabel}
      >
        {buttonContent}
      </button>
    </div>
  )
}
