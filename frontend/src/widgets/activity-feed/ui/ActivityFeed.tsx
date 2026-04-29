import type { DashboardEvent } from '@/entities/event/model/types'
import type { LogEntry } from '@/entities/log/model/types'

type Props = { logs: LogEntry[]; events: DashboardEvent[] }

export function ActivityFeed({ logs, events }: Props) {
  const recentLogs = logs.slice(0, 5)
  const auditLogs = logs.slice(5, 9)

  return (
    <section className="context-panel">
      <div className="context-brand">
        <div className="context-dot" />
        <strong>Live activity</strong>
      </div>

      <div className="context-group">
        <h4>Recent logs</h4>
        <div className="context-list">
          {recentLogs.length === 0 ? (
            <article className="context-mini context-empty-note">
              <p>Awaiting live backend logs.</p>
            </article>
          ) : (
            recentLogs.map((log) => (
              <article key={log.id} className="context-item context-item-card">
                <span className="context-item-icon">{log.level.slice(0, 1).toUpperCase()}</span>
                <span>
                  <strong>{log.level.toUpperCase()}</strong> · {log.message}
                </span>
              </article>
            ))
          )}
        </div>
      </div>

      <div className="context-group">
        <h4>Event stream</h4>
        <div className="context-list">
          {events.slice(0, 4).map((event) => (
            <article key={event.id} className="context-item context-item-active context-item-card">
              <span className="context-item-icon">▣</span>
              <span>{event.title}</span>
            </article>
          ))}
        </div>
      </div>

      <div className="context-group">
        <h4>Audit</h4>
        <div className="context-list compact-list">
          {auditLogs.length === 0 ? (
            <article className="context-mini context-empty-note">
              <p>No audit logs yet.</p>
            </article>
          ) : (
            auditLogs.map((log) => (
              <article key={log.id} className="context-mini">
                <span>{log.level.toUpperCase()} · {log.at.slice(11, 19)}</span>
                <p>{log.message}</p>
              </article>
            ))
          )}
        </div>
      </div>
    </section>
  )
}
