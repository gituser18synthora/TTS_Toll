import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the UI runs on Vite (5173) and proxies /api to the FastAPI backend.
const apiTarget = process.env.PORTAL_API || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: process.env.VITE_HOST || '127.0.0.1',
    port: Number(process.env.VITE_PORT || 5173),
    strictPort: true,
    proxy: { '/api': { target: apiTarget } },
  },
})
