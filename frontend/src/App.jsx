import { useEffect, useState } from 'react'
import { api } from './api.js'
import { navigate, useHashRoute, usePoll } from './hooks.js'
import { Badge, ConfigCtx, ToastProvider, fmtHours } from './components/ui.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import UploadPage from './pages/UploadPage.jsx'
import SourcesPage from './pages/SourcesPage.jsx'
import ReviewPage from './pages/ReviewPage.jsx'
import ExportPage from './pages/ExportPage.jsx'

const TABS = [
  { path: 'dashboard', label: 'Dashboard' },
  { path: 'upload', label: 'Upload' },
  { path: 'sources', label: 'Files & Jobs' },
  { path: 'review', label: 'Review' },
  { path: 'export', label: 'Export' },
]

function WorkerPill({ health, error }) {
  if (error) return <Badge color="red" title={error}>backend offline</Badge>
  if (!health) return null
  const w = health.worker
  const engines = Object.entries(w.engines || {}).map(([k, v]) => `${k}: ${v}`).join('\n')
  return w.online ? (
    <Badge color="green" title={`A worker process is running and will pick up jobs${engines ? `\n\n${engines}` : ''}`}>
      worker online · {w.devices.join(', ')}
    </Badge>
  ) : (
    <Badge color="amber" title="Start it with run_dev.sh or: cd backend && python -m app.worker">
      worker offline: jobs will wait
    </Badge>
  )
}

export default function App() {
  const route = useHashRoute()
  const [config, setConfig] = useState(null)
  const health = usePoll(() => api.health(), 10000)
  const stats = usePoll(() => api.stats(), (d) => (d && d.jobs.running + d.jobs.queued > 0 ? 3000 : 10000), [route.path])

  useEffect(() => {
    api.config().then(setConfig).catch(() => setTimeout(() => window.location.reload(), 3000))
  }, [])

  const s = stats.data
  const page = {
    dashboard: <DashboardPage />,
    upload: <UploadPage />,
    sources: <SourcesPage query={route.query} />,
    review: <ReviewPage query={route.query} />,
    export: <ExportPage />,
  }[route.path] || <DashboardPage />

  return (
    <ConfigCtx.Provider value={config}>
      <ToastProvider>
        <div className="flex h-full flex-col">
          <header className="flex items-center gap-6 border-b border-slate-200 bg-white px-4">
            <div className="flex items-center gap-2 py-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-indigo-600 text-white">
                <svg viewBox="0 0 32 32" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M5 16h3l2-7 3 14 3-10 2 6 2-3h7" /></svg>
              </div>
              <div className="leading-tight">
                <div className="text-sm font-semibold text-slate-800">Orpheus Dataset Portal</div>
                <div className="text-[11px] text-slate-500">TTS data preparation</div>
              </div>
            </div>
            <nav className="flex gap-1 self-stretch">
              {TABS.map((t) => {
                const active = route.path === t.path || (t.path === 'dashboard' && !TABS.some((x) => x.path === route.path))
                return (
                  <button
                    key={t.path}
                    onClick={() => navigate(t.path)}
                    className={`border-b-2 px-3 text-sm font-medium ${active ? 'border-indigo-600 text-indigo-700' : 'border-transparent text-slate-500 hover:text-slate-800'}`}
                  >
                    {t.label}
                    {t.path === 'review' && s?.needs_review > 0 && (
                      <span className="ml-1.5 rounded-full bg-amber-100 px-1.5 text-[11px] text-amber-800">{s.needs_review}</span>
                    )}
                  </button>
                )
              })}
            </nav>
            <div className="ml-auto flex items-center gap-4 text-xs text-slate-500">
              {s && (
                <span className="hidden md:inline">
                  approved <b className="text-emerald-700">{fmtHours(s.by_status.approved?.hours || 0)}</b> · pending{' '}
                  <b className="text-slate-700">{fmtHours(s.by_status.pending?.hours || 0)}</b>
                  {(s.jobs.running > 0 || s.jobs.queued > 0) && (
                    <>
                      {' '}· jobs <b className="text-sky-700">{s.jobs.running} running, {s.jobs.queued} queued</b>
                    </>
                  )}
                </span>
              )}
              <WorkerPill health={health.data} error={health.error} />
            </div>
          </header>
          <main className="min-h-0 flex-1 overflow-auto">{config ? page : <div className="p-8 text-slate-500">Connecting to backend…</div>}</main>
        </div>
      </ToastProvider>
    </ConfigCtx.Provider>
  )
}
