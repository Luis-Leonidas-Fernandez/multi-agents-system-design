import { useMemo } from 'react'
import type { Agent } from '@/entities/agent/model/types'
import { SelectAgent } from '@/features/select-agent/ui/SelectAgent'
import { useSendAgentAction } from '@/features/send-agent-action/model/useSendAgentAction'
import { AgentWorkflow } from '@/widgets/agent-workflow/ui/AgentWorkflow'
import { ActivityFeed } from '@/widgets/activity-feed/ui/ActivityFeed'
import { AgentComposer } from '@/widgets/agent-composer/ui/AgentComposer'
import { MetricsSidebar } from '@/widgets/metrics-sidebar/ui/MetricsSidebar'
import { useDashboardStream } from '@/pages/dashboard/model/useDashboardStream'

const AGENTS: Agent[] = [
  { id: 'analysis', name: 'Analysis', status: 'running' },
  { id: 'web', name: 'Web Scraper', status: 'idle' },
  { id: 'math', name: 'Math', status: 'success' },
]

export function DashboardPage() {
  const { selectedAgent, selectedAgentId, setSelectedAgentId, reasoning, conclusion, finalResponse, events, logs, tokens, connected, mode, sendAction } = useDashboardStream(AGENTS)
  const { message, setMessage, send } = useSendAgentAction((text) => sendAction({ agentId: selectedAgent.id, message: text }))

  const feedEvents = useMemo(() => events.slice(0, 25), [events])

  return (
    <main>
      <header>
        <SelectAgent agents={AGENTS} selectedAgentId={selectedAgentId} onSelect={setSelectedAgentId} />
        <p>{connected ? `Connected (${mode})` : 'Disconnected'}</p>
      </header>
      <section>
        <ActivityFeed events={feedEvents} />
        <div>
          <AgentWorkflow reasoning={reasoning} conclusion={conclusion} finalResponse={finalResponse} />
          <AgentComposer value={message} onChange={setMessage} onSend={send} />
        </div>
        <MetricsSidebar tokens={tokens} logs={logs.slice(0, 10)} />
      </section>
    </main>
  )
}
