import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { Agent } from '@/entities/agent/model/types'
import { useSendAgentAction } from '@/features/send-agent-action/model/useSendAgentAction'
import { AgentWorkflow } from '@/widgets/agent-workflow/ui/AgentWorkflow'
import { ActivityFeed } from '@/widgets/activity-feed/ui/ActivityFeed'
import { AgentComposer } from '@/widgets/agent-composer/ui/AgentComposer'
import { MetricsSidebar } from '@/widgets/metrics-sidebar/ui/MetricsSidebar'
import { useDashboardStream } from '@/pages/dashboard/model/useDashboardStream'

const AGENTS: Agent[] = [
  { id: 'analysis', name: 'Analysis', status: 'running' },
  { id: 'web', name: 'Web Scraper', status: 'idle' },
  { id: 'math', name: 'Math', status: 'success' },
]

const NAV = [
  { label: 'Dashboard', icon: <GridIcon /> },
  { label: 'Agents', icon: <BotIcon /> },
  { label: 'Console', icon: <TerminalIcon /> },
  { label: 'Data', icon: <DatabaseIcon /> },
  { label: 'Settings', icon: <FolderGearIcon /> },
]

function HeaderIcon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {children}
    </svg>
  )
}

function BellIcon() {
  return (
    <HeaderIcon>
      <path d="M12 22a2.5 2.5 0 0 0 2.45-2h-4.9A2.5 2.5 0 0 0 12 22Zm6-6V11a6 6 0 1 0-12 0v5l-2 2v1h16v-1l-2-2Z" fill="currentColor" />
    </HeaderIcon>
  )
}

function GearIcon() {
  return (
    <HeaderIcon>
      <path d="M19.14 12.94a7.43 7.43 0 0 0 .05-.94 7.43 7.43 0 0 0-.05-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7.12 7.12 0 0 0-1.62-.94l-.36-2.54a.5.5 0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54c-.58.22-1.12.53-1.62.94l-2.39-.96a.5.5 0 0 0-.6.22L2.7 8.84a.5.5 0 0 0 .12.64l2.03 1.58c-.03.31-.05.62-.05.94s.02.63.05.94L2.82 14.52a.5.5 0 0 0-.12.64l1.92 3.32c.13.23.4.33.6.22l2.39-.96c.5.4 1.04.72 1.62.94l.36 2.54c.03.24.24.42.5.42h3.84c.26 0 .47-.18.5-.42l.36-2.54c.58-.22 1.12-.53 1.62-.94l2.39.96c.2.11.47.01.6-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58ZM12 15.2A3.2 3.2 0 1 1 12 8.8a3.2 3.2 0 0 1 0 6.4Z" fill="currentColor" />
    </HeaderIcon>
  )
}

function HelpIcon() {
  return (
    <HeaderIcon>
      <path d="M11 18h2v-2h-2v2Zm1-16a7 7 0 0 0-4.95 11.95l1.4-1.4A5 5 0 1 1 16 10c0 2.08-1.28 2.8-2.28 3.36-.9.5-1.72.96-1.72 2.64v.5h2v-.5c0-.5.11-.7.97-1.19C16.12 13.95 18 12.91 18 10a7 7 0 0 0-6-8Z" fill="currentColor" />
    </HeaderIcon>
  )
}

function UtilityIcon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {children}
    </svg>
  )
}

function GridIcon() {
  return (
    <UtilityIcon>
      <path d="M4 4h7v7H4V4Zm9 0h7v7h-7V4ZM4 13h7v7H4v-7Zm9 0h7v7h-7v-7Z" fill="currentColor" />
    </UtilityIcon>
  )
}

function BotIcon() {
  return (
    <UtilityIcon>
      <path d="M12 3a3 3 0 0 0-3 3v1H7a2 2 0 0 0-2 2v6a5 5 0 0 0 5 5h4a5 5 0 0 0 5-5V9a2 2 0 0 0-2-2h-2V6a3 3 0 0 0-3-3Zm-1 4V6a1 1 0 0 1 2 0v1h-2Zm-2 4h2v2H9v-2Zm6 0h-2v2h2v-2Zm-3 7c-2.2 0-4-.9-4-2h8c0 1.1-1.8 2-4 2Z" fill="currentColor" />
    </UtilityIcon>
  )
}

function TerminalIcon() {
  return (
    <UtilityIcon>
      <path d="M4 5h16v14H4V5Zm2 2v10h12V7H6Zm2 2 2 2-2 2 1.5 1.5L13 11l-3.5-3.5L8 9Zm6 5h4v2h-4v-2Z" fill="currentColor" />
    </UtilityIcon>
  )
}

function DatabaseIcon() {
  return (
    <UtilityIcon>
      <path d="M12 3c-4.4 0-8 1.3-8 3s3.6 3 8 3 8-1.3 8-3-3.6-3-8-3Zm0 8c-4.4 0-8-1.3-8-3v4c0 1.7 3.6 3 8 3s8-1.3 8-3v-4c0 1.7-3.6 3-8 3Zm0 6c-4.4 0-8-1.3-8-3v4c0 1.7 3.6 3 8 3s8-1.3 8-3v-4c0 1.7-3.6 3-8 3Z" fill="currentColor" />
    </UtilityIcon>
  )
}

function FolderGearIcon() {
  return (
    <UtilityIcon>
      <path d="M10 4 8.6 2H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h7.1a6.5 6.5 0 0 1-.1-1.2 6.6 6.6 0 0 1 2-4.7V7a2 2 0 0 0-2-2h-1Zm10.2 11.7.6-.5-1-1.7-.8.3a4.5 4.5 0 0 0-.7-.4l-.1-.8h-2l-.1.8c-.2.1-.5.2-.7.4l-.8-.3-1 1.7.6.5c0 .2 0 .5 0 .7l-.6.5 1 1.7.8-.3c.2.2.4.3.7.4l.1.8h2l.1-.8c.2-.1.5-.2.7-.4l.8.3 1-1.7-.6-.5c0-.2 0-.5 0-.7ZM18 19a2 2 0 1 1 0-4 2 2 0 0 1 0 4Z" fill="currentColor" />
    </UtilityIcon>
  )
}

export function DashboardPage() {
  const { selectedAgent, reasoning, conclusion, finalResponse, events, logs, tokens, sessionId, turnId, turnLatencyMs, messageCount, lastUserMessage, lastAssistantResponse, connected, mode, sendAction, abortAction } = useDashboardStream(AGENTS)
  const [isThinking, setIsThinking] = useState(false)
  const [phase, setPhase] = useState<'idle' | 'thinking' | 'responding' | 'error'>('idle')
  const { message, setMessage, send } = useSendAgentAction((text) => sendAction({ agentId: selectedAgent.id, message: text }))

  const feedLogs = useMemo(() => logs.slice(0, 20), [logs])

  const abort = () => {
    abortAction({ reason: 'user aborted from dashboard' })
    setIsThinking(false)
    setPhase('idle')
  }

  useEffect(() => {
    if (!isThinking) return
    if (reasoning || conclusion || finalResponse) {
      setPhase('responding')
      const timer = window.setTimeout(() => {
        setIsThinking(false)
        setPhase('idle')
      }, 900)
      return () => window.clearTimeout(timer)
    }
  }, [isThinking, reasoning, conclusion, finalResponse])

  useEffect(() => {
    if (mode !== 'websocket') return

    if (connected) {
      if (phase === 'error') setPhase('idle')
      return
    }

    if (phase !== 'idle') return

    setPhase('error')
    const timer = window.setTimeout(() => setPhase('idle'), 1800)
    return () => window.clearTimeout(timer)
  }, [connected, mode, phase])

  return (
    <div className="dashboard-app">
      <header className="top-app-bar">
        <div className="top-app-brand">
          <div className="brand-mark">A</div>
          <div>
            <strong>AgenticOrchestrator</strong>
            <small>Realtime control room</small>
          </div>
        </div>

        <label className="top-search">
          <span>⌕</span>
          <input placeholder="Search systems..." />
        </label>

        <div className="top-actions">
          <button type="button" aria-label="Notifications"><BellIcon /></button>
          <button type="button" aria-label="Settings"><GearIcon /></button>
          <button type="button" aria-label="Help"><HelpIcon /></button>
          <div className="avatar">L</div>
        </div>
      </header>

      <div className="dashboard-frame">
        <nav className="utility-rail" aria-label="Main navigation">
          <button type="button" className="utility-btn active" aria-label="Dashboard"><GridIcon /></button>
          {NAV.slice(1).map((item) => (
            <button key={item.label} type="button" className="utility-btn" aria-label={item.label}>{item.icon}</button>
          ))}
          <div className="utility-bottom">
            <div className="avatar avatar-small">P</div>
          </div>
        </nav>

        <aside className="context-sidebar">
          <ActivityFeed logs={feedLogs} events={events} />
        </aside>

        <main className="workspace">
          <div className="workspace-topline">
            <div className="status-pills">
              <span className={`status-pill ${connected ? 'status-online' : 'status-offline'}`}>{connected ? 'System Online' : 'System Offline'}</span>
              <span className="status-pill">Mode {mode === 'websocket' ? 'WebSocket' : 'Mock'}</span>
              <span className="status-pill">Session {sessionId || '—'}</span>
            </div>
          </div>

          <div className="workspace-hero">
            <img className="workspace-hero-image" src="/hero-ai.jpeg" alt="AI" />
            <h1>How can I help you today?</h1>
          </div>

          <div className="chat-stage">
            <AgentWorkflow reasoning={reasoning} conclusion={conclusion} finalResponse={finalResponse} isThinking={isThinking} status={phase} />
          </div>

          <div className="workspace-composer-shell">
            <AgentComposer
              value={message}
              onChange={setMessage}
              status={phase}
              onSend={() => {
                if (!message.trim()) return
                setIsThinking(true)
                setPhase('thinking')
                send()
              }}
              onAbort={abort}
            />
          </div>

        </main>

        <aside className="inspector-panel">
          <MetricsSidebar
            tokens={tokens}
            sessionId={sessionId}
            connected={connected}
            mode={mode}
            events={events}
            selectedAgent={selectedAgent}
            messageCount={messageCount}
            lastUserMessage={lastUserMessage}
            turnId={turnId}
            turnLatencyMs={turnLatencyMs}
            lastAssistantResponse={lastAssistantResponse}
          />
        </aside>
      </div>
    </div>
  )
}
