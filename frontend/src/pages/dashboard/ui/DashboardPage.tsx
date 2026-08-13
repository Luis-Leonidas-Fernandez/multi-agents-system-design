import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { Agent } from '@/entities/agent/model/types'
import { useSendAgentAction } from '@/features/send-agent-action/model/useSendAgentAction'
import { AgentWorkflow } from '@/widgets/agent-workflow/ui/AgentWorkflow'
import { ActivityFeed } from '@/widgets/activity-feed/ui/ActivityFeed'
import { AgentComposer } from '@/widgets/agent-composer/ui/AgentComposer'
import { MetricsSidebar } from '@/widgets/metrics-sidebar/ui/MetricsSidebar'
import { useDashboardStream } from '@/pages/dashboard/model/useDashboardStream'
import jellyFrame1 from '../../../assets/jellyfish/frame-1.png'
import jellyFrame2 from '../../../assets/jellyfish/frame-2.png'
import jellyFrame3 from '../../../assets/jellyfish/frame-3.png'
import jellyFrame4 from '../../../assets/jellyfish/frame-4.png'
import jellyFrame5 from '../../../assets/jellyfish/frame-5.png'
import jellyFrame6 from '../../../assets/jellyfish/frame-6.png'
import jellyFrame7 from '../../../assets/jellyfish/frame-7.png'

type DashboardView = 'chat' | 'activity' | 'runtime'

const AGENTS: Agent[] = [
  { id: 'analysis', name: 'Analysis', status: 'running' },
  { id: 'web', name: 'Web Scraper', status: 'idle' },
  { id: 'math', name: 'Math', status: 'success' },
]

const NAV: Array<{ label: string; icon: ReactNode; view?: DashboardView }> = [
  { label: 'Dashboard', icon: <GridIcon />, view: 'chat' },
  { label: 'Activity', icon: <TerminalIcon />, view: 'activity' },
  { label: 'Runtime', icon: <DatabaseIcon />, view: 'runtime' },
  { label: 'Settings', icon: <FolderGearIcon /> },
]

const JELLYFISH_FRAMES = [jellyFrame1, jellyFrame2, jellyFrame3, jellyFrame4, jellyFrame5, jellyFrame6, jellyFrame7]

function truncateTitle(text: string, maxChars = 25) {
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars).trimEnd()}...`
}

function UtilityIcon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {children}
    </svg>
  )
}

function JellyfishMascot() {
  const [currentFrame, setCurrentFrame] = useState(0)

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setCurrentFrame((prev) => (prev + 1) % JELLYFISH_FRAMES.length)
    }, 170)

    return () => window.clearInterval(intervalId)
  }, [])

  return (
    <div className="jellyfish-mascot" aria-hidden="true">
      <div className="jellyfish-drift">
        <div className="jellyfish-follow">
          <img className="jellyfish-img" src={JELLYFISH_FRAMES[currentFrame]} alt="" />
        </div>
      </div>
    </div>
  )
}

function GridIcon() {
  return (
    <UtilityIcon>
      <path d="M4.5 10.2 12 4l7.5 6.2V19a1.5 1.5 0 0 1-1.5 1.5h-3.6v-5.2h-4.8v5.2H6A1.5 1.5 0 0 1 4.5 19v-8.8Z" fill="currentColor" />
      <path d="M3.4 10.8 12 3.8l8.6 7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </UtilityIcon>
  )
}

function TerminalIcon() {
  return (
    <UtilityIcon>
      <path
        d="M3.5 12h3.1l1.7-4.2 2.6 9 2.4-6.1 1.7 3.3h5.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="3.5" cy="12" r="1.25" fill="currentColor" />
      <circle cx="20.5" cy="12" r="1.25" fill="currentColor" />
    </UtilityIcon>
  )
}

function DatabaseIcon() {
  return (
    <UtilityIcon>
      <path
        d="M12 3.15a8.85 8.85 0 1 1-6.92 3.28"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.45"
        strokeLinecap="round"
      />
      <path
        d="M3.25 9.15V3.55h5.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.45"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 6.75v5.65l-3.55 3.55" fill="none" stroke="currentColor" strokeWidth="2.45" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M16.9 7.2h.01M19.05 12h.01M16.9 16.8h.01M12 19.05h.01M7.1 16.8h.01M4.95 12h.01" fill="none" stroke="currentColor" strokeWidth="2.95" strokeLinecap="round" />
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
  const summaryTitle = truncateTitle('¿Trabajamos juntos Luis?')
  const {
    selectedAgent,
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
    artifacts,
    moodleAuditTree,
    connected,
    mode,
    sendAction,
    sendArtifactAction,
    abortAction,
  } = useDashboardStream(AGENTS)
  const [isThinking, setIsThinking] = useState(false)
  const [phase, setPhase] = useState<'idle' | 'thinking' | 'responding' | 'error'>('idle')
  const [googleCalendarEnabled, setGoogleCalendarEnabled] = useState(false)
  const [activeView, setActiveView] = useState<DashboardView>('chat')
  const [pendingUserMessage, setPendingUserMessage] = useState('')
  const composerAnchorRef = useRef<HTMLDivElement | null>(null)
  const { message, setMessage } = useSendAgentAction((text) => sendAction({ agentId: selectedAgent.id, message: text }))

  useEffect(() => {
    if (!pendingUserMessage) return
    if (lastUserMessage && lastUserMessage.trim() === pendingUserMessage.trim()) {
      setPendingUserMessage('')
      return
    }
    if (finalResponse) {
      setPendingUserMessage('')
    }
  }, [pendingUserMessage, lastUserMessage, finalResponse])

  useEffect(() => {
    if (activeView !== 'chat') return
    if (!finalResponse && artifacts.length === 0 && !moodleAuditTree) return
    window.requestAnimationFrame(() => {
      composerAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    })
  }, [activeView, finalResponse, artifacts.length, moodleAuditTree])

  const feedLogs = useMemo(() => logs.slice(0, 20), [logs])
  const activeAgents = AGENTS.filter((agent) => agent.status === 'running').length
  const latestEvent = events[0]
  const latestArtifact = artifacts[0]
  const activeAgentMeta = selectedAgent.status === 'running' ? 'activo' : selectedAgent.status
  const overviewCards = [
    {
      label: 'AGENTE ACTIVO',
      value: selectedAgent.name,
      meta: activeAgentMeta,
      tone: selectedAgent.status,
    },
    {
      label: 'Estado actual',
      value: turnLatencyMs ? `${turnLatencyMs}ms` : 'esperando',
      meta: turnId ? `Turn ${turnId}` : 'turno en espera',
      tone: phase === 'error' ? 'error' : 'running',
    },
    {
      label: 'artefactos',
      value: String(artifacts.length),
      meta: latestArtifact ? latestArtifact.jsonPath.split('/').pop() ?? latestArtifact.jsonPath : 'archivos generados',
      tone: artifacts.length > 0 ? 'success' : 'idle',
    },
    {
      label: 'sistema',
      value: connected ? 'estable' : 'degradado',
      meta: `${activeAgents}. activo -${messageCount} mensajes`,
      tone: connected ? 'success' : 'error',
    },
  ] as const

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

  const renderChatView = () => (
    <section className="dashboard-focus">
      <section className="dashboard-summary-shell">
        <section className="dashboard-summary-block">
          <header className="dashboard-summary-title">
            <div className="dashboard-summary-mascot-shell">
              <JellyfishMascot />
            </div>
            <div className="dashboard-summary-title-inner">
              <h2 title="¿Trabajamos juntos Luis?">{summaryTitle}</h2>
            </div>
          </header>
          <section className="overview-grid overview-grid-minimal" aria-label="Operational overview">
            {overviewCards.map((card) => (
              <article key={card.label} className={`overview-card overview-card-${card.tone}`}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
                <small>{card.meta}</small>
              </article>
            ))}
          </section>
        </section>
      </section>

      <section className="dashboard-chat-shell">
        <section className="chat-hub">
          <div className="chat-hub-body">
            <div className="chat-stage chat-stage-unified">
              <AgentWorkflow
                lastUserMessage={pendingUserMessage || lastUserMessage}
                reasoning={reasoning}
                conclusion={conclusion}
                finalResponse={finalResponse}
                artifacts={artifacts}
                moodleAuditTree={moodleAuditTree}
                googleCalendarEnabled={googleCalendarEnabled}
                onApproveArtifact={(artifactPath) => sendArtifactAction({ kind: 'approve', artifactPath })}
                onDeleteArtifact={(artifactPath) => sendArtifactAction({ kind: 'delete', artifactPath })}
                onCreateEventsFromArtifact={(artifactPath) => sendArtifactAction({ kind: 'create_events', artifactPath, enabledMcps: googleCalendarEnabled ? ['google_calendar'] : [] })}
                isThinking={isThinking}
                status={phase}
              />
            </div>
          </div>
        </section>

        <div className="dashboard-composer-shell" ref={composerAnchorRef}>
          <div className="composer-inline-shell">
            <AgentComposer
              value={message}
              onChange={setMessage}
              status={phase}
              googleCalendarEnabled={googleCalendarEnabled}
              onToggleGoogleCalendar={() => setGoogleCalendarEnabled((current) => !current)}
              onSend={() => {
                if (!message.trim()) return
                const nextMessage = message.trim()
                setPendingUserMessage(nextMessage)
                setIsThinking(true)
                setPhase('thinking')
                sendAction({
                  agentId: selectedAgent.id,
                  message: nextMessage,
                  enabledMcps: googleCalendarEnabled ? ['google_calendar'] : [],
                })
                setMessage('')
              }}
              onAbort={abort}
            />
          </div>
        </div>
      </section>
    </section>
  )

  const renderActivityView = () => (
    <section className="workspace-section workspace-page-shell">
      <div className="workspace-section-head">
        <div>
          <span className="workspace-kicker">Activity page</span>
          <h2>Live activity + logs</h2>
        </div>
      </div>
      <div className="workspace-page-grid">
        <ActivityFeed logs={feedLogs} events={events} />
      </div>
    </section>
  )

  const renderRuntimeView = () => (
    <section className="workspace-section workspace-page-shell">
      <div className="workspace-section-head">
        <div>
          <span className="workspace-kicker">Runtime page</span>
          <h2>Runtime source of truth</h2>
        </div>
      </div>
      <div className="workspace-page-grid">
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
      </div>
    </section>
  )

  return (
    <div className="dashboard-app">
      <div className={`dashboard-frame dashboard-frame-${activeView}`}>
        <nav className="utility-rail" aria-label="Main navigation">
          {NAV.map((item) => (
            <button
              key={item.label}
              type="button"
              className={`utility-btn ${item.view === activeView ? 'active' : ''}`}
              aria-label={item.label}
              onClick={() => {
                if (!item.view) return
                setActiveView(item.view)
              }}
            >
              {item.icon}
            </button>
          ))}
          <div className="utility-bottom">
            <div className="avatar avatar-small">P</div>
          </div>
        </nav>

        <main className="workspace">
          {activeView === 'chat' ? renderChatView() : null}
          {activeView === 'activity' ? renderActivityView() : null}
          {activeView === 'runtime' ? renderRuntimeView() : null}
        </main>
      </div>
    </div>
  )
}
