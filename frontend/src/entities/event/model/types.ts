export type EventKind = 'success' | 'error' | 'info' | 'warning' | 'token' | 'trace' | 'action'

export type DashboardEvent = {
  id: string
  kind: EventKind
  title: string
  detail: string
  at: string
  agentId: string
}
