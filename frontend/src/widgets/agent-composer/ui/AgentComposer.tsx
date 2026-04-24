import { SendAgentAction } from '@/features/send-agent-action/ui/SendAgentAction'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
}

export function AgentComposer({ value, onChange, onSend }: Props) {
  return <SendAgentAction value={value} onChange={onChange} onSend={onSend} />
}
