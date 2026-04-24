type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
}

export function SendAgentAction({ value, onChange, onSend }: Props) {
  return (
    <div>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} placeholder="Ask the agent..." />
      <button onClick={onSend}>Send</button>
    </div>
  )
}
