import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Self-hosted Indic fonts (no CDN): every reviewer sees the same glyphs; each script's
// font file is only downloaded when text in that script is on screen.
import '@fontsource/noto-sans-devanagari/400.css'
import '@fontsource/noto-sans-devanagari/600.css'
import '@fontsource/noto-sans-tamil/400.css'
import '@fontsource/noto-sans-tamil/600.css'
import '@fontsource/noto-sans-telugu/400.css'
import '@fontsource/noto-sans-telugu/600.css'
import '@fontsource/noto-sans-malayalam/400.css'
import '@fontsource/noto-sans-malayalam/600.css'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
