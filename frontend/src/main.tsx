import { createRoot } from 'react-dom/client'
import { App } from '@/app/App'
import '@/shared/styles/global.css'

const root = document.getElementById('root')

if (!root) throw new Error('Root element not found')

createRoot(root).render(<App />)
