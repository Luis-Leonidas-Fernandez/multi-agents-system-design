export type LogEntry = {
  id: string
  level: 'debug' | 'info' | 'warn' | 'error'
  message: string
  at: string
}
