import { useEffect, useMemo, useState } from 'react'
import type { Agent } from '@/entities/agent/model/types'
import type { DashboardEvent } from '@/entities/event/model/types'
import type { LogEntry } from '@/entities/log/model/types'
import type { TokenMetric } from '@/entities/token-metric/model/types'
import { createDashboardRealtimeClient } from '@/shared/api/realtime'
import { WS_URL } from '@/shared/config/env'
import type { DashboardRealtimeMessage } from '@/shared/types/realtime'

const INITIAL_AGENT_ID = 'analysis'

export function useDashboardStream(agents: Agent[]) {
  const client = useMemo(() => createDashboardRealtimeClient(WS_URL || undefined), [])
  const [selectedAgentId, setSelectedAgentId] = useState(INITIAL_AGENT_ID)
  const [reasoning, setReasoning] = useState('')
  const [conclusion, setConclusion] = useState('')
  const [finalResponse, setFinalResponse] = useState('')
  const [events, setEvents] = useState<DashboardEvent[]>([])
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [tokens, setTokens] = useState<TokenMetric>({ prompt: 0, completion: 0, total: 0 })
  const [sessionId, setSessionId] = useState('')
  const [turnId, setTurnId] = useState('')
  const [turnLatencyMs, setTurnLatencyMs] = useState(0)
  const [messageCount, setMessageCount] = useState(0)
  const [lastUserMessage, setLastUserMessage] = useState('')
  const [lastAssistantResponse, setLastAssistantResponse] = useState('')
  const [connected, setConnected] = useState(false)
  const [mode, setMode] = useState<'mock' | 'websocket'>('mock')

  const selectedAgent = useMemo(
    () => agents.find((agent) => agent.id === selectedAgentId) ?? agents[0],
    [agents, selectedAgentId],
  )

  useEffect(() => {
    const unsubscribe = client.subscribe((message: DashboardRealtimeMessage) => {
      if (message.type === 'snapshot') {
        setSelectedAgentId(message.payload.activeAgent.id)
        setSessionId(message.payload.sessionId)
        setReasoning(message.payload.reasoning)
        setConclusion(message.payload.conclusion)
        setFinalResponse(message.payload.finalResponse)
        setTurnId(message.payload.turnId)
        setTurnLatencyMs(message.payload.turnLatencyMs)
        setMessageCount(message.payload.messageCount)
        setLastUserMessage(message.payload.lastUserMessage)
        setLastAssistantResponse(message.payload.lastAssistantResponse)
        setEvents((current) => [...message.payload.events, ...current].slice(0, 50))
        setLogs((current) => [...message.payload.logs, ...current].slice(0, 50))
        setTokens(message.payload.tokens)
      }
      if (message.type === 'event') setEvents((current) => [message.payload, ...current].slice(0, 50))
      if (message.type === 'log') setLogs((current) => [message.payload, ...current].slice(0, 50))
      if (message.type === 'tokens') setTokens(message.payload)
      if (message.type === 'reasoning') {
        setReasoning(message.payload.reasoning)
        setConclusion(message.payload.conclusion)
        setFinalResponse(message.payload.finalResponse)
      }
      if (message.type === 'status') {
        setConnected(message.payload.connected)
        setMode(message.payload.mode)
      }
    })

    client.connect()
    return () => {
      unsubscribe()
      client.disconnect()
    }
  }, [client])

  return {
    reasoning,
    conclusion,
    finalResponse,
    events,
    logs,
    tokens,
    sessionId,
    turnId,
    turnLatencyMs,
    messageCount,
    lastUserMessage,
    lastAssistantResponse,
    connected,
    mode,
    selectedAgent,
    selectedAgentId,
    setSelectedAgentId,
    sendAction: client.sendAction,
    abortAction: client.abortAction,
  }
}
