import type { Agent } from '@/entities/agent/model/types'
import type { DashboardEvent } from '@/entities/event/model/types'
import type { LogEntry } from '@/entities/log/model/types'
import type { TokenMetric } from '@/entities/token-metric/model/types'

export type DashboardSnapshot = {
  activeAgent: Agent
  reasoning: string
  conclusion: string
  finalResponse: string
  turnId: string
  turnLatencyMs: number
  messageCount: number
  lastUserMessage: string
  lastAssistantResponse: string
  events: DashboardEvent[]
  logs: LogEntry[]
  tokens: TokenMetric
  sessionId: string
}

export type DashboardAction = {
  agentId: string
  message: string
}

export type DashboardAbort = {
  reason?: string
}

export type DashboardRealtimeMessage =
  | { type: 'snapshot'; payload: DashboardSnapshot }
  | { type: 'event'; payload: DashboardEvent }
  | { type: 'log'; payload: LogEntry }
  | { type: 'tokens'; payload: TokenMetric }
  | { type: 'reasoning'; payload: { reasoning: string; conclusion: string; finalResponse: string } }
  | { type: 'status'; payload: { connected: boolean; mode: 'mock' | 'websocket' } }
