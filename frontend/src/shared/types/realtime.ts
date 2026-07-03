import type { Agent } from '@/entities/agent/model/types'
import type { DashboardEvent } from '@/entities/event/model/types'
import type { LogEntry } from '@/entities/log/model/types'
import type { TokenMetric } from '@/entities/token-metric/model/types'

export type DashboardArtifact = {
  jsonPath: string
  markdownPath: string
  structureValid: boolean
  approved: boolean
  validCount: number
  invalidCount: number
  issueCount: number
  issues: Array<{ assignment_title: string; severity: string; message: string } | Record<string, string>>
  jsonContent: string
  markdownContent: string
  syncCreatedCount: number
}

export type MoodleAuditTreeNode = {
  id: string
  kind:
    | 'course'
    | 'page'
    | 'section'
    | 'forum'
    | 'quiz'
    | 'assignment'
    | 'document'
    | 'image'
    | 'video'
    | 'preview'
    | 'link'
    | 'google_slides'
    | 'google_drive'
    | 'external_redirect'
    | 'unknown'
  title: string
  url?: string
  canonicalUrl?: string
  previewUrl?: string
  downloadUrl?: string
  redirectUrl?: string
  mimeType?: string
  subtitle?: string
  description?: string
  badges?: string[]
  metadata?: Record<string, string | number | boolean>
  children: MoodleAuditTreeNode[]
}

export type MoodleAuditTree = {
  jobUid: string
  courseName: string
  auditPath: string
  summaryPath?: string
  stats: {
    pageCount: number
    retainedPageCount?: number
    externalRedirectCount?: number
    downloadDocumentCount?: number
    assignmentLikeCount?: number
    resourceTypeCounts?: Record<string, number>
  }
  root: MoodleAuditTreeNode
}

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
  artifacts: DashboardArtifact[]
  moodleAuditTree?: MoodleAuditTree | null
}

export type DashboardAction = {
  agentId: string
  message: string
  enabledMcps?: string[]
}

export type DashboardArtifactAction = {
  kind: 'approve' | 'delete' | 'create_events'
  artifactPath: string
  enabledMcps?: string[]
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
