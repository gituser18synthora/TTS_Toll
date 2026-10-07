import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

// ---------- formatting ----------
export function fmtDuration(sec) {
  if (sec === null || sec === undefined) return '–'
  const s = Math.round(sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`
}

export function fmtClock(sec) {
  if (sec === null || sec === undefined) return '–'
  const m = Math.floor(sec / 60)
  const s = (sec - m * 60).toFixed(1).padStart(4, '0')
  return `${m}:${s}`
}

export const fmtHours = (h) =>
  h === null || h === undefined ? '–' : h === 0 ? '0 min' : h < 1 / 60 ? `${Math.max(1, Math.round(h * 3600))} s` : h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(2)} h`

export function fmtBytes(n) {
  if (!n && n !== 0) return '–'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024
    i++
  }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`
}

export function fmtTime(iso) {
  if (!iso) return '–'
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

// ---------- primitives ----------
const BTN = {
  primary: 'bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300',
  secondary: 'bg-white text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 disabled:text-slate-400',
  success: 'bg-emerald-600 text-white hover:bg-emerald-700 disabled:bg-emerald-300',
  danger: 'bg-rose-600 text-white hover:bg-rose-700 disabled:bg-rose-300',
  ghost: 'text-slate-600 hover:bg-slate-200/70 disabled:text-slate-300',
}

export function Button({ variant = 'secondary', size = 'md', className = '', ...props }) {
  const sz = size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3 py-1.5 text-sm'
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors disabled:cursor-not-allowed ${sz} ${BTN[variant]} ${className}`}
      {...props}
    />
  )
}

const BADGE = {
  slate: 'bg-slate-100 text-slate-700 ring-slate-300',
  green: 'bg-emerald-50 text-emerald-700 ring-emerald-300',
  red: 'bg-rose-50 text-rose-700 ring-rose-300',
  amber: 'bg-amber-50 text-amber-800 ring-amber-300',
  blue: 'bg-sky-50 text-sky-700 ring-sky-300',
  indigo: 'bg-indigo-50 text-indigo-700 ring-indigo-300',
}

export function Badge({ color = 'slate', children, className = '', title }) {
  return (
    <span title={title} className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset ${BADGE[color]} ${className}`}>
      {children}
    </span>
  )
}

export const STATUS_COLOR = {
  pending: 'slate', approved: 'green', rejected: 'red',
  uploaded: 'slate', processing: 'blue', done: 'green', failed: 'red', needs_speakers: 'amber',
  queued: 'slate', running: 'blue', cancelled: 'amber',
}

export function StatusBadge({ status }) {
  return <Badge color={STATUS_COLOR[status] || 'slate'}>{status}</Badge>
}

export function ProgressBar({ value, className = '', color = 'bg-indigo-500' }) {
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-slate-200 ${className}`}>
      <div className={`h-full rounded-full transition-[width] duration-500 ${color}`} style={{ width: `${Math.max(0, Math.min(100, value || 0))}%` }} />
    </div>
  )
}

export function Card({ title, actions, children, className = '' }) {
  return (
    <section className={`rounded-lg bg-white shadow-sm ring-1 ring-slate-200 ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-2.5">
          <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Field({ label, children, hint }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-slate-500">{hint}</span>}
    </label>
  )
}

// width-less base (combine with w-auto / w-44 …) and the usual full-width variant
export const inputBase =
  'rounded-md border-0 bg-white px-2.5 py-1.5 text-sm text-slate-800 ring-1 ring-slate-300 placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500 focus:outline-none disabled:bg-slate-50 disabled:text-slate-500'
export const inputCls = `w-full ${inputBase}`

export function Modal({ title, onClose, children, footer, wide = false }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose?.()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div data-modal className="fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 pt-[8vh]" onMouseDown={onClose}>
      <div className={`w-full ${wide ? 'max-w-4xl' : 'max-w-lg'} rounded-lg bg-white shadow-xl`} onMouseDown={(e) => e.stopPropagation()}>
        <header className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
          <h3 className="font-semibold text-slate-800">{title}</h3>
          <button className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="p-4">{children}</div>
        {footer && <footer className="flex justify-end gap-2 border-t border-slate-100 px-4 py-3">{footer}</footer>}
      </div>
    </div>
  )
}

export function ConfirmDialog({ title, message, confirmLabel = 'Confirm', danger = false, onConfirm, onCancel }) {
  const [busy, setBusy] = useState(false)
  return (
    <Modal
      title={title}
      onClose={onCancel}
      footer={
        <>
          <Button onClick={onCancel}>Cancel</Button>
          <Button
            variant={danger ? 'danger' : 'primary'}
            disabled={busy}
            onClick={async () => {
              setBusy(true)
              try {
                await onConfirm()
              } finally {
                setBusy(false)
              }
            }}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="text-sm text-slate-600">{message}</div>
    </Modal>
  )
}

// ---------- toasts ----------
const ToastCtx = createContext(() => {})
export const useToast = () => useContext(ToastCtx)

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  const id = useRef(0)
  const push = useCallback((message, kind = 'info') => {
    const t = { id: ++id.current, message, kind }
    setToasts((ts) => [...ts.slice(-3), t])
    setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== t.id)), kind === 'error' ? 7000 : 3000)
  }, [])
  const colors = { info: 'bg-slate-800', success: 'bg-emerald-700', error: 'bg-rose-700' }
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-2">
        {toasts.map((t) => (
          <div key={t.id} className={`pointer-events-auto max-w-xl rounded-md px-4 py-2 text-sm text-white shadow-lg ${colors[t.kind]}`}>
            {t.message}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  )
}

// ---------- app-wide config (languages, reject reasons, ...) ----------
export const ConfigCtx = createContext(null)
export const useConfig = () => useContext(ConfigCtx)

export function languageName(config, code) {
  return config?.languages?.find((l) => l.code === code)?.name || code
}
