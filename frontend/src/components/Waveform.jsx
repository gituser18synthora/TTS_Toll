import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { fmtClock } from './ui.jsx'

// One WaveSurfer instance for the lifetime of the component; `url` changes just load new audio.
const Waveform = forwardRef(function Waveform({ url, autoplay = false, rate = 1 }, ref) {
  const container = useRef(null)
  const ws = useRef(null)
  const autoplayRef = useRef(autoplay)
  autoplayRef.current = autoplay
  const rateRef = useRef(rate)
  rateRef.current = rate
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const w = WaveSurfer.create({
      container: container.current,
      height: 88,
      waveColor: '#a5b4fc',
      progressColor: '#4f46e5',
      cursorColor: '#0f172a',
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      dragToSeek: true,
    })
    w.on('ready', (d) => {
      setLoading(false)
      setDuration(d)
      w.setPlaybackRate(rateRef.current, true)
      if (autoplayRef.current) w.play().catch(() => {})
    })
    w.on('play', () => setPlaying(true))
    w.on('pause', () => setPlaying(false))
    w.on('finish', () => setPlaying(false))
    w.on('timeupdate', (t) => setTime(t))
    w.on('error', (e) => {
      setLoading(false)
      setError(String(e?.message || e))
    })
    ws.current = w
    return () => w.destroy()
  }, [])

  useEffect(() => {
    const w = ws.current
    if (!w || !url) return
    setLoading(true)
    setError(null)
    setTime(0)
    w.load(url).catch((e) => {
      if (e?.name !== 'AbortError') setError(String(e?.message || e))
    })
  }, [url])

  useEffect(() => {
    ws.current?.setPlaybackRate(rate, true)
  }, [rate])

  useImperativeHandle(ref, () => ({
    playPause: () => ws.current?.playPause().catch(() => {}),
    restart: () => {
      ws.current?.setTime(0)
      ws.current?.play().catch(() => {})
    },
    stop: () => ws.current?.stop(),
    skip: (s) => ws.current?.skip(s),
  }))

  return (
    <div className="rounded-md bg-slate-50 p-2 ring-1 ring-slate-200">
      <div ref={container} className={loading ? 'opacity-40' : ''} />
      <div className="mt-1 flex items-center gap-3 text-xs text-slate-500">
        <button
          type="button"
          tabIndex={-1}
          onClick={() => ws.current?.playPause().catch(() => {})}
          className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-indigo-600 text-white hover:bg-indigo-700"
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="currentColor"><rect x="5" y="4" width="3.5" height="12" rx="1" /><rect x="11.5" y="4" width="3.5" height="12" rx="1" /></svg>
          ) : (
            <svg viewBox="0 0 20 20" className="ml-0.5 h-3.5 w-3.5" fill="currentColor"><path d="M6 4.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 6 4.5Z" /></svg>
          )}
        </button>
        <span className="font-mono tabular-nums">
          {fmtClock(time)} / {fmtClock(duration)}
        </span>
        {loading && <span>loading…</span>}
        {error && <span className="text-rose-600">audio error: {error}</span>}
      </div>
    </div>
  )
})

export default Waveform
