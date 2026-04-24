import { useState } from 'react'

export function useSendAgentAction(onAction: (message: string) => void) {
  const [message, setMessage] = useState('')

  const send = () => {
    const trimmed = message.trim()
    if (!trimmed) return
    onAction(trimmed)
    setMessage('')
  }

  return { message, setMessage, send }
}
