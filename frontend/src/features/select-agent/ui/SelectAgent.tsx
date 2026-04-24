import type { Agent } from '@/entities/agent/model/types'

type Props = {
  agents: Agent[]
  selectedAgentId: string
  onSelect: (agentId: string) => void
}

export function SelectAgent({ agents, selectedAgentId, onSelect }: Props) {
  return (
    <div>
      <label>Agent</label>
      <select value={selectedAgentId} onChange={(e) => onSelect(e.target.value)}>
        {agents.map((agent) => (
          <option key={agent.id} value={agent.id}>
            {agent.name}
          </option>
        ))}
      </select>
    </div>
  )
}
