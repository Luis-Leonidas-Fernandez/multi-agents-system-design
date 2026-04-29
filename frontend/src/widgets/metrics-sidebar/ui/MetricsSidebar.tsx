import type { DashboardEvent } from '@/entities/event/model/types'
import type { Agent } from '@/entities/agent/model/types'

type Props = {
  tokens: { prompt: number; completion: number; total: number }
  sessionId: string
  connected: boolean
  mode: 'mock' | 'websocket'
  events: DashboardEvent[]
  selectedAgent: Agent
  messageCount: number
  lastUserMessage: string
  turnId: string
  turnLatencyMs: number
  lastAssistantResponse: string
}

export function MetricsSidebar({ tokens, sessionId, connected, mode, events, selectedAgent, messageCount, lastUserMessage, turnId, turnLatencyMs, lastAssistantResponse }: Props) {
  return (
    <section className="inspector-shell">
      <div className="inspector-head inspector-head-tight">
        <div>
          <h3>Runtime</h3>
          <p>Source of truth</p>
        </div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Session overview</span>
        <div className="health-card">
          <div className="health-row">
            <span>Status</span>
            <strong>{connected ? `Online (${mode})` : 'Offline'}</strong>
          </div>
          <div className="health-row">
            <span>Session</span>
            <strong>{sessionId || '—'}</strong>
          </div>
          <div className="health-row">
            <span>Messages</span>
            <strong>{messageCount}</strong>
          </div>
          <div className="health-row">
            <span>Agent</span>
            <strong>{selectedAgent.name}</strong>
          </div>
          <div className="health-row">
            <span>Agent status</span>
            <strong>{selectedAgent.status}</strong>
          </div>
        </div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Turn telemetry</span>
        <div className="inspector-note">
          <div className="health-row">
            <span>Turn</span>
            <strong>{turnId || '—'}</strong>
          </div>
          <div className="health-row">
            <span>Latency</span>
            <strong>{turnLatencyMs ? `${turnLatencyMs}ms` : '—'}</strong>
          </div>
          <div className="health-row">
            <span>Assistant</span>
            <strong>{lastAssistantResponse ? 'Ready' : 'Waiting'}</strong>
          </div>
        </div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Last assistant response</span>
        <div className="inspector-note">{lastAssistantResponse || 'Waiting for the first answer...'}</div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Last user input</span>
        <div className="inspector-note">{lastUserMessage || 'Waiting for the first question...'}</div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Tokens</span>
        <div className="inspector-metrics">
          <div className="metric-tile">
            <span>Prompt</span>
            <strong>{tokens.prompt}</strong>
          </div>
          <div className="metric-tile">
            <span>Completion</span>
            <strong>{tokens.completion}</strong>
          </div>
          <div className="metric-tile">
            <span>Total</span>
            <strong>{tokens.total}</strong>
          </div>
        </div>
      </div>

      <div className="inspector-section">
        <span className="section-label">Latest audit</span>
        <div className="inspector-audit">
          {events.length === 0 ? (
            <p className="empty-state">No audit entries yet.</p>
          ) : (
            events.slice(0, 3).map((event) => (
              <article key={event.id} className={`audit-row audit-${event.kind}`}>
                <div className="event-meta">
                  <strong>{event.kind}</strong>
                  <span>{event.at}</span>
                </div>
                <p className="event-title">{event.title}</p>
                <small>{event.detail}</small>
              </article>
            ))
          )}
        </div>
      </div>
      <div className="inspector-footer">Web dashboard is the primary source of truth</div>
    </section>
  )
}
