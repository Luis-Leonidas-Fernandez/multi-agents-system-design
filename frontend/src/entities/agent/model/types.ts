export type Agent = {
  id: string
  name: string
  status: 'idle' | 'running' | 'success' | 'error'
}
