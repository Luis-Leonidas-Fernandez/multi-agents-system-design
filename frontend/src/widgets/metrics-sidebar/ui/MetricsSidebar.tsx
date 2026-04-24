type Props = {
  tokens: { prompt: number; completion: number; total: number }
  logs: { id: string; level: string; message: string; at: string }[]
}

export function MetricsSidebar({ tokens, logs }: Props) {
  return (
    <aside>
      <h2>Tokens</h2>
      <p>{tokens.total}</p>
      <h2>Logs</h2>
      {logs.map((log) => (
        <p key={log.id}>{log.level}: {log.message}</p>
      ))}
    </aside>
  )
}
