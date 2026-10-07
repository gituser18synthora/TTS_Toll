import { useCallback, useEffect, useRef, useState } from 'react'

// --- hash routing: "#/review?status=pending" -> { path: "review", query: { status: "pending" } }
function parseHash() {
  const raw = window.location.hash.replace(/^#\/?/, '')
  const [path, search = ''] = raw.split('?')
  return { path: path || 'dashboard', query: Object.fromEntries(new URLSearchParams(search)) }
}

export function navigate(path, query = {}) {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null && v !== '') sp.set(k, v)
  const s = sp.toString()
  window.location.hash = `/${path}${s ? `?${s}` : ''}`
}

export function useHashRoute() {
  const [route, setRoute] = useState(parseHash)
  useEffect(() => {
    const onChange = () => setRoute(parseHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

// --- polling fetch: re-runs `fn` every `interval(data)` ms while mounted
export function usePoll(fn, interval, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const intervalRef = useRef(interval)
  intervalRef.current = interval
  const timer = useRef(null)
  const alive = useRef(true)

  const load = useCallback(async () => {
    clearTimeout(timer.current)
    try {
      const d = await fnRef.current()
      if (!alive.current) return d
      setData(d)
      setError(null)
      const ms = typeof intervalRef.current === 'function' ? intervalRef.current(d) : intervalRef.current
      if (ms) timer.current = setTimeout(load, ms)
      return d
    } catch (e) {
      if (!alive.current) return null
      setError(e.message)
      timer.current = setTimeout(load, 5000)
      return null
    }
  }, [])

  useEffect(() => {
    alive.current = true
    load()
    return () => {
      alive.current = false
      clearTimeout(timer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, reload: load, setData }
}

// --- persisted UI preference (per browser)
export function useLocalSetting(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(`portal.${key}`)
      return raw === null ? initial : JSON.parse(raw)
    } catch {
      return initial
    }
  })
  useEffect(() => {
    try {
      localStorage.setItem(`portal.${key}`, JSON.stringify(value))
    } catch {
      /* storage unavailable */
    }
  }, [key, value])
  return [value, setValue]
}
