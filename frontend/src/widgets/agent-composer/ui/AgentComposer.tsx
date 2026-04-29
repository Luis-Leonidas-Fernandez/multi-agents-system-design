import { SendAgentAction } from '@/features/send-agent-action/ui/SendAgentAction'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAbort: () => void
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

export function AgentComposer({ value, onChange, onSend, onAbort, status }: Props) {
  return <SendAgentAction value={value} onChange={onChange} onSend={onSend} onAbort={onAbort} status={status} />
}
