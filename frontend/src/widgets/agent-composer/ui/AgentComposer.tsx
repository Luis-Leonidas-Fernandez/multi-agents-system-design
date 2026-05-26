import { SendAgentAction } from '@/features/send-agent-action/ui/SendAgentAction'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAbort: () => void
  googleCalendarEnabled: boolean
  onToggleGoogleCalendar: () => void
  status: 'idle' | 'thinking' | 'responding' | 'error'
}

export function AgentComposer({
  value,
  onChange,
  onSend,
  onAbort,
  googleCalendarEnabled,
  onToggleGoogleCalendar,
  status,
}: Props) {
  return (
    <SendAgentAction
      value={value}
      onChange={onChange}
      onSend={onSend}
      onAbort={onAbort}
      googleCalendarEnabled={googleCalendarEnabled}
      onToggleGoogleCalendar={onToggleGoogleCalendar}
      status={status}
    />
  )
}
