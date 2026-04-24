import type { DashboardEvent } from '@/entities/event/model/types'

type Props = { events: DashboardEvent[] }

export function ActivityFeed({ events }: Props) {
  return (
    <aside>
      {events.map((event) => (
        <article key={event.id}>
          <strong>{event.kind}</strong>
          <p>{event.title}</p>
          <small>{event.detail}</small>
        </article>
      ))}
    </aside>
  )
}
