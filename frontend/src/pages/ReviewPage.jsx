import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'
import { navigate, useLocalSetting } from '../hooks.js'
import Waveform from '../components/Waveform.jsx'
import WordDiff from '../components/WordDiff.jsx'
import { Badge, Button, StatusBadge, fmtClock, inputBase, inputCls, languageName, useConfig, useToast } from '../components/ui.jsx'

const FLAG_INFO = {
  forced_cut: 'cut inside continuous speech (no pause long enough) — check the edges',
  very_quiet: 'very low recording level before normalization',
  low_asr_confidence: 'ASR was unsure',
  asr_no_speech: 'ASR thinks there may be no speech',
  asr_repetition: 'ASR output repeats itself (possible hallucination)',
  empty_transcript: 'ASR returned nothing',
  sparse_transcript: 'transcript much shorter than the audio suggests',
  no_asr: 'no ASR engine was available — type the transcript',
  asr_disagreement: 'the two ASR systems disagree (WER above the language threshold)',
  numbers_expanded: 'digits were spelled out: check they match how the number was read',
  digits_remaining: 'some digits could not be spelled out: write them as words',
  chars_removed: 'unsupported characters (emoji, symbols, other scripts) were removed',
  wrong_script: 'ASR wrote the wrong script (e.g. Urdu for Hindi): type the transcript',
  annotation_removed: 'ASR wrote a non-speech tag like [music]: the clip may contain it',
  borderline_quality: 'DNSMOS quality between the reject and review thresholds',
  music: 'music or singing detected',
  babble: 'background talk / crowd detected (possible overlapping speech)',
  speaker_mismatch: "this voice doesn't match the speaker's other clips (wrong speaker label?)",
  duplicate: 'same recording as another clip (see "duplicate of")',
  overlap: 'two speakers talk at the same time in part of this clip',
}

const DEFAULT_FILTERS = { status: 'pending', sort: 'position', order: 'asc', page_size: '50' }

// does a (possibly just edited) clip still belong to the current filtered list?
function matchesFilter(clip, f) {
  if (f.status && f.status !== 'all' && clip.status !== f.status) return false
  if (f.needs_review === 'true' && !clip.needs_review) return false
  if (f.needs_review === 'false' && clip.needs_review) return false
  return true
}

function isTypingTarget(t) {
  if (!t) return false
  const tag = t.tagName
  if (tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable) return true
  return tag === 'INPUT' && !['checkbox', 'radio', 'button', 'submit', 'range'].includes(t.type)
}

function Metric({ label, value, warn = false, title }) {
  return (
    <div title={title}>
      <div className="text-[11px] tracking-wide text-slate-400 uppercase">{label}</div>
      <div className={`font-mono text-sm tabular-nums ${warn ? 'text-amber-700' : 'text-slate-700'}`}>{value ?? '–'}</div>
    </div>
  )
}

function Shortcuts() {
  const items = [
    ['Space', 'play / pause'], ['A', 'approve'], ['R', 'reject'], ['Enter', 'save & next'], ['← →', 'prev / next'],
    ['E', 'edit transcript'], ['Ctrl+Enter', 'approve & next (also while typing)'], ['Esc', 'leave text box'], ['P', 'replay'],
    ['O', 'original / denoised'],
  ]
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
      {items.map(([k, v]) => (
        <span key={k}>
          <kbd>{k}</kbd> {v}
        </span>
      ))}
    </div>
  )
}

export default function ReviewPage({ query }) {
  const config = useConfig()
  const toast = useToast()
  const filters = useMemo(() => ({ ...DEFAULT_FILTERS, ...query }), [query])
  const filterKey = JSON.stringify(filters)

  const [speakers, setSpeakers] = useState([])
  const [sources, setSources] = useState([])
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sel, setSel] = useState(0)
  const [checked, setChecked] = useState(() => new Set())
  const [draft, setDraft] = useState('')
  const [saveState, setSaveState] = useState('idle') // idle | dirty | saving | saved | error
  const [search, setSearch] = useState(filters.q || '')
  const [variant, setVariant] = useState('clip') // clip | original (before denoising)

  const [autoAdvance, setAutoAdvance] = useLocalSetting('autoAdvance', true)
  const [autoplay, setAutoplay] = useLocalSetting('autoplay', true)
  const [rate, setRate] = useLocalSetting('rate', 1)
  const [rejectReason, setRejectReason] = useLocalSetting('rejectReason', 'manual')

  const waveRef = useRef(null)
  const textRef = useRef(null)
  const listRef = useRef(null)
  const pending = useRef(null) // unsaved transcript edit: { id, text }
  const saveTimer = useRef(null)

  const items = data?.items || []
  const current = items[sel] || null
  const pageSize = Number(filters.page_size) || 50

  useEffect(() => {
    api.speakers().then(setSpeakers).catch(() => {})
    api.sources().then(setSources).catch(() => {})
  }, [])

  const setFilter = (patch) => navigate('review', { ...filters, ...patch })

  const load = useCallback(
    async (offset, selectLast = false) => {
      setLoading(true)
      try {
        const res = await api.clips({ ...filters, offset: Math.max(0, offset), page_size: pageSize })
        setData(res)
        setSel(selectLast ? Math.max(0, res.items.length - 1) : 0)
        setChecked(new Set())
        return res
      } catch (e) {
        toast(e.message, 'error')
        return null
      } finally {
        setLoading(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [filterKey],
  )

  useEffect(() => {
    setSearch(filters.q || '')
    load(0)
  }, [load, filters.q])

  // when the selected clip changes: reset the editor and keep the row visible
  useEffect(() => {
    setDraft(current?.text || '')
    setSaveState('idle')
    setVariant('clip')
    listRef.current?.querySelector(`[data-clip="${current?.id}"]`)?.scrollIntoView({ block: 'nearest' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id])

  const replaceItem = (clip) => setData((d) => (d ? { ...d, items: d.items.map((c) => (c.id === clip.id ? clip : c)) } : d))

  // ---------------- transcript auto-save ----------------
  async function flushSave() {
    clearTimeout(saveTimer.current)
    const p = pending.current
    if (!p) return true
    pending.current = null
    setSaveState('saving')
    try {
      const clip = await api.updateClip(p.id, { text: p.text })
      replaceItem(clip)
      setSaveState(pending.current ? 'dirty' : 'saved')
      return true
    } catch (e) {
      pending.current = pending.current || p
      setSaveState('error')
      toast(`Save failed: ${e.message}`, 'error')
      return false
    }
  }

  function onDraftChange(text) {
    setDraft(text)
    if (!current) return
    pending.current = { id: current.id, text }
    setSaveState('dirty')
    clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(flushSave, 700)
  }

  // ---------------- navigation ----------------
  async function select(i) {
    await flushSave()
    setSel(i)
  }

  // `updated`: a clip just changed by this handler (state updates are not visible in this closure yet)
  async function next(updated) {
    if (!data) return
    if (sel < items.length - 1) return select(sel + 1)
    await flushSave()
    const rows = updated ? items.map((c) => (c.id === updated.id ? updated : c)) : items
    const left = rows.filter((c) => !matchesFilter(c, filters)).length // reviewed rows that drop out of the list
    const nextOffset = data.offset + rows.length - left
    if (nextOffset >= data.total - left) {
      if (left) {
        toast('End of list. Showing the clips that still match the filters.', 'info')
        load(0)
      } else toast('End of list', 'info')
      return
    }
    load(nextOffset)
  }

  async function prev() {
    if (!data) return
    if (sel > 0) return select(sel - 1)
    if (data.offset > 0) {
      await flushSave()
      load(data.offset - pageSize, true)
    }
  }

  // ---------------- review actions ----------------
  async function review(status, { advance = autoAdvance } = {}) {
    if (!current) return
    const id = current.id
    clearTimeout(saveTimer.current)
    const body = { status }
    const edit = pending.current?.id === id ? pending.current.text : draft !== (current.text || '') ? draft : undefined
    pending.current = null
    if (edit !== undefined) body.text = edit
    if (status === 'rejected') body.reject_reason = rejectReason
    try {
      const clip = await api.updateClip(id, body)
      replaceItem(clip)
      setSaveState('saved')
      if (advance) await next(clip)
    } catch (e) {
      if (edit !== undefined) pending.current = { id, text: edit }
      toast(e.message, 'error')
    }
  }

  async function saveAndNext() {
    await flushSave()
    next()
  }

  async function bulk(action) {
    const ids = [...checked]
    if (!ids.length) return
    try {
      const res = await api.bulk({ ids, action, reject_reason: action === 'reject' ? rejectReason : undefined })
      toast(`${res.updated} clip(s) updated${res.skipped ? `, ${res.skipped} skipped (${res.skipped_reason})` : ''}`, 'success')
      load(data.offset)
    } catch (e) {
      toast(e.message, 'error')
    }
  }

  // ---------------- keyboard ----------------
  const onKey = useRef(null)
  onKey.current = (e) => {
    if (e.isComposing || e.keyCode === 229) return // IME composition (Indic input methods)
    if (document.querySelector('[data-modal]')) return
    const t = e.target
    const mod = e.ctrlKey || e.metaKey
    if (mod && e.key === 'Enter') {
      e.preventDefault()
      if (t === textRef.current) t.blur()
      review('approved', { advance: true })
      return
    }
    if (isTypingTarget(t)) {
      if (t === textRef.current) {
        if (e.key === 'Escape') {
          e.preventDefault()
          t.blur()
        } else if (e.key === 'Enter' && !e.shiftKey && !e.altKey) {
          e.preventDefault() // transcripts are single-line: Enter commits
          t.blur()
          saveAndNext()
        }
      }
      return
    }
    if (mod || e.altKey) return
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key
    const actions = {
      ' ': () => waveRef.current?.playPause(),
      a: () => review('approved'),
      r: () => review('rejected'),
      Enter: () => saveAndNext(),
      ArrowRight: () => next(),
      ArrowLeft: () => prev(),
      p: () => waveRef.current?.restart(),
      o: () => current?.denoised && setVariant((v) => (v === 'clip' ? 'original' : 'clip')),
      e: () => {
        const el = textRef.current
        if (el) {
          el.focus()
          el.setSelectionRange(el.value.length, el.value.length)
        }
      },
    }
    if (actions[k]) {
      e.preventDefault()
      actions[k]()
    }
  }
  useEffect(() => {
    const handler = (e) => onKey.current(e)
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  useEffect(() => () => clearTimeout(saveTimer.current), [])

  // ---------------- render ----------------
  const allChecked = items.length > 0 && items.every((c) => checked.has(c.id))
  const noFocus = (e) => e.preventDefault() // keep focus off buttons so shortcuts keep working
  const sourceName = (id) => sources.find((s) => s.id === Number(id))?.original_filename

  return (
    <div className="flex h-full flex-col">
      {/* filters */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 text-sm">
        <select className={`${inputBase} w-auto`} value={filters.status} onChange={(e) => setFilter({ status: e.target.value })}>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="all">All statuses</option>
        </select>
        <select className={`${inputBase} w-auto`} value={filters.needs_review || ''} onChange={(e) => setFilter({ needs_review: e.target.value })}>
          <option value="">Flagged: any</option>
          <option value="true">Flagged only</option>
          <option value="false">Not flagged</option>
        </select>
        <select className={`${inputBase} w-auto`} value={filters.speaker_id || ''} onChange={(e) => setFilter({ speaker_id: e.target.value })}>
          <option value="">All speakers</option>
          {speakers.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <select className={`${inputBase} w-auto`} value={filters.language || ''} onChange={(e) => setFilter({ language: e.target.value })}>
          <option value="">All languages</option>
          {config.languages.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
        </select>
        <select className={`${inputBase} w-auto max-w-56`} value={filters.source_file_id || ''} onChange={(e) => setFilter({ source_file_id: e.target.value })}>
          <option value="">All files</option>
          {sources.map((s) => <option key={s.id} value={s.id}>#{s.id} {s.original_filename}</option>)}
        </select>
        <select className={`${inputBase} w-auto`} value={filters.flag || ''} onChange={(e) => setFilter({ flag: e.target.value })}>
          <option value="">Any flag</option>
          {Object.keys(FLAG_INFO).map((f) => <option key={f} value={f}>{f}</option>)}
        </select>
        <form
          className="flex"
          onSubmit={(e) => {
            e.preventDefault()
            setFilter({ q: search })
          }}
        >
          <input className={`${inputBase} w-44`} placeholder="search transcript…" value={search} onChange={(e) => setSearch(e.target.value)} />
        </form>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-slate-500">sort</span>
          <select className={`${inputBase} w-auto`} value={filters.sort} onChange={(e) => setFilter({ sort: e.target.value })}>
            <option value="position">position in file</option>
            <option value="asr_confidence">ASR confidence</option>
            <option value="duration">duration</option>
            <option value="quality_score">quality score</option>
            <option value="asr_agreement">ASR WER</option>
            <option value="updated_at">last edited</option>
          </select>
          <Button size="sm" variant="ghost" onClick={() => setFilter({ order: filters.order === 'asc' ? 'desc' : 'asc' })}>
            {filters.order === 'asc' ? '↑ asc' : '↓ desc'}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate('review', DEFAULT_FILTERS)}>reset</Button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* clip list */}
        <div className="flex w-[42%] min-w-[380px] flex-col border-r border-slate-200 bg-white">
          {checked.size > 0 ? (
            <div className="flex items-center gap-2 border-b border-indigo-100 bg-indigo-50 px-3 py-1.5 text-xs">
              <b className="text-indigo-800">{checked.size} selected</b>
              <Button size="sm" variant="success" onClick={() => bulk('approve')}>Approve</Button>
              <Button size="sm" variant="danger" onClick={() => bulk('reject')}>Reject ({rejectReason})</Button>
              <Button size="sm" onClick={() => bulk('pending')}>Reset to pending</Button>
              <Button size="sm" variant="ghost" onClick={() => setChecked(new Set())}>clear</Button>
            </div>
          ) : (
            <div className="flex items-center justify-between border-b border-slate-100 px-3 py-1.5 text-xs text-slate-500">
              <span>
                {data ? (data.total ? `${data.offset + 1}–${data.offset + items.length} of ${data.total}` : 'no clips match') : '…'}
                {filters.source_file_id && sourceName(filters.source_file_id) && <> · {sourceName(filters.source_file_id)}</>}
              </span>
              {loading && <span>loading…</span>}
            </div>
          )}
          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto">
            <table className="w-full table-fixed text-xs">
              <thead className="sticky top-0 z-10 bg-slate-50 text-left text-[11px] text-slate-500">
                <tr>
                  <th className="w-8 px-2 py-1.5">
                    <input
                      type="checkbox"
                      checked={allChecked}
                      onChange={() => setChecked(allChecked ? new Set() : new Set(items.map((c) => c.id)))}
                      title="select all on this page"
                    />
                  </th>
                  <th className="w-12 py-1.5 font-medium">#</th>
                  <th className="w-12 py-1.5 text-right font-medium">dur</th>
                  <th className="w-12 py-1.5 text-right font-medium">conf</th>
                  <th className="px-2 py-1.5 font-medium">transcript</th>
                  <th className="w-16 py-1.5 font-medium">status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((c, i) => (
                  <tr
                    key={c.id}
                    data-clip={c.id}
                    onClick={() => select(i)}
                    className={`cursor-pointer border-b border-slate-100 ${i === sel ? 'bg-indigo-50 outline-1 -outline-offset-1 outline-indigo-300' : 'hover:bg-slate-50'}`}
                  >
                    <td className="px-2 py-1.5" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={checked.has(c.id)}
                        onChange={() =>
                          setChecked((s) => {
                            const n = new Set(s)
                            n.has(c.id) ? n.delete(c.id) : n.add(c.id)
                            return n
                          })
                        }
                      />
                    </td>
                    <td className="py-1.5 text-slate-400 tabular-nums">
                      {c.source_file_id}·{c.idx}
                    </td>
                    <td className="py-1.5 text-right tabular-nums">{c.duration.toFixed(1)}</td>
                    <td className={`py-1.5 text-right tabular-nums ${c.asr_confidence !== null && c.asr_confidence < -0.6 ? 'text-amber-700' : 'text-slate-500'}`}>
                      {c.asr_confidence?.toFixed(2) ?? '–'}
                    </td>
                    <td className="truncate px-2 py-1.5" dir="auto" title={c.text || ''}>
                      {c.needs_review && <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-amber-500 align-middle" title={c.flags.join(', ')} />}
                      {c.text || <i className="text-slate-400">{c.asr_primary_text === null ? 'transcribing…' : 'empty'}</i>}
                    </td>
                    <td className="py-1.5">
                      <StatusBadge status={c.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between border-t border-slate-200 px-3 py-1.5 text-xs">
            <Button size="sm" variant="ghost" disabled={!data || data.offset === 0} onClick={() => load(data.offset - pageSize)}>
              ‹ prev page
            </Button>
            <select className="rounded border-0 bg-transparent text-xs text-slate-500" value={filters.page_size} onChange={(e) => setFilter({ page_size: e.target.value })}>
              {[25, 50, 100, 200].map((n) => <option key={n} value={n}>{n} / page</option>)}
            </select>
            <Button size="sm" variant="ghost" disabled={!data || data.offset + items.length >= data.total} onClick={() => load(data.offset + items.length)}>
              next page ›
            </Button>
          </div>
        </div>

        {/* detail */}
        <div className="min-w-0 flex-1 overflow-y-auto p-4">
          {!current ? (
            <div className="mt-16 text-center text-sm text-slate-500">
              {data && !data.total ? 'Nothing to review with these filters.' : 'Select a clip.'}
            </div>
          ) : (
            <div className="mx-auto max-w-4xl space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold text-slate-800">Clip {current.id}</h2>
                <StatusBadge status={current.status} />
                {current.reject_reason && <Badge color="red">{current.reject_reason}</Badge>}
                {current.text_edited && <Badge color="indigo">edited</Badge>}
                <span className="text-xs text-slate-500">
                  {current.speaker_name} ({current.gender}){current.speaker_label ? ` [${current.speaker_label}]` : ''} · {languageName(config, current.language)} · #{current.source_file_id}{' '}
                  {current.source_filename} @ {fmtClock(current.start_time)}–{fmtClock(current.end_time)}
                </span>
              </div>

              <Waveform ref={waveRef} url={variant === 'original' ? `/api/clips/${current.id}/audio?variant=original` : current.audio_url} autoplay={autoplay} rate={rate} />
              {current.denoised && (
                <div className="-mt-1 flex items-center gap-2 text-xs text-slate-500">
                  <span>Listening to</span>
                  <div className="flex rounded-md bg-slate-100 p-0.5 ring-1 ring-slate-200">
                    {[['clip', 'denoised (exported)'], ['original', 'original']].map(([v, label]) => (
                      <button key={v} onMouseDown={noFocus} onClick={() => setVariant(v)}
                        className={`rounded px-2 py-0.5 ${variant === v ? 'bg-white font-medium text-slate-800 shadow-sm' : 'text-slate-500'}`}>
                        {label}
                      </button>
                    ))}
                  </div>
                  <span><kbd>O</kbd> toggles. Check that denoising did not damage the voice.</span>
                </div>
              )}

              <div>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-600">
                    Transcript <span className="font-normal text-slate-400">(exported as “{current.speaker_name}: …”)</span>
                  </span>
                  <span className={{ error: 'text-rose-600', dirty: 'text-amber-600', saving: 'text-slate-500', saved: 'text-emerald-600' }[saveState] || 'text-slate-400'}>
                    {{ error: 'save failed', dirty: 'unsaved…', saving: 'saving…', saved: 'saved ✓' }[saveState] || ''}
                  </span>
                </div>
                <textarea
                  ref={textRef}
                  dir="auto"
                  lang={current.language === 'hinglish' ? 'hi' : current.language}
                  rows={3}
                  value={draft}
                  onChange={(e) => onDraftChange(e.target.value)}
                  onBlur={() => flushSave()}
                  className={`transcript ${inputCls} resize-y`}
                  placeholder="Type what is said…"
                />
              </div>

              <div className="rounded-md bg-white p-3 text-sm ring-1 ring-slate-200">
                <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                  <span>
                    ASR primary <span className="text-slate-400">({current.asr_primary_engine || 'whisper'})</span> — differences from your transcript
                  </span>
                  {current.primary_normalized && draft !== current.primary_normalized && (
                    <button className="text-indigo-600 hover:underline" onMouseDown={noFocus} onClick={() => onDraftChange(current.primary_normalized)}
                      title={current.primary_normalized}>
                      use primary
                    </button>
                  )}
                </div>
                <div className="leading-7" lang={current.language === 'hinglish' ? 'hi' : current.language}>
                  {current.asr_primary_text === null ? <i className="text-slate-400">not transcribed yet</i> : <WordDiff from={current.asr_primary_text} to={draft} />}
                </div>
                {current.asr_secondary_text !== null && (
                  <>
                    <div className="mt-3 mb-1 flex items-center justify-between text-xs text-slate-500">
                      <span>
                        ASR secondary <span className="text-slate-400">({current.asr_secondary_engine || '?'})</span> — differences from primary
                      </span>
                      {current.secondary_normalized && draft !== current.secondary_normalized && (
                        <button className="text-indigo-600 hover:underline" onMouseDown={noFocus} onClick={() => onDraftChange(current.secondary_normalized)}
                          title={current.secondary_normalized}>
                          use secondary
                        </button>
                      )}
                    </div>
                    <div className="leading-7" lang={current.language === 'hinglish' ? 'hi' : current.language}>
                      <WordDiff from={current.asr_primary_text} to={current.asr_secondary_text} />
                    </div>
                  </>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3 rounded-md bg-white p-3 ring-1 ring-slate-200 sm:grid-cols-6">
                <Metric label="duration" value={`${current.duration.toFixed(2)} s`} />
                <Metric label="ASR conf" value={current.asr_confidence?.toFixed(2)} warn={current.asr_confidence !== null && current.asr_confidence < -0.6} title="mean log-probability of the primary ASR (closer to 0 = more confident)" />
                <Metric label="ASR WER" value={current.asr_agreement !== null ? `${(current.asr_agreement * 100).toFixed(0)}%` : null} warn={current.flags.includes('asr_disagreement')} title="word error rate between the two ASR outputs (0% = identical)" />
                <Metric
                  label={current.quality_info?.method === 'snr' ? 'quality (SNR)' : 'quality'}
                  value={current.quality_score?.toFixed(2)}
                  warn={current.quality_score !== null && config.quality && current.quality_score < config.quality.review_below}
                  title={current.quality_info ? Object.entries(current.quality_info).map(([k, v]) => `${k}: ${v}`).join('  ') : 'DNSMOS OVRL, 1-5'}
                />
                <Metric label="input level" value={current.loudness_lufs !== null ? `${current.loudness_lufs.toFixed(1)} LUFS` : null} warn={current.loudness_lufs !== null && current.loudness_lufs < -45} />
                <Metric label="voice match" value={current.speaker_similarity !== null ? current.speaker_similarity.toFixed(2) : null}
                  warn={current.flags.includes('speaker_mismatch')} title="cosine similarity to this speaker's voice centroid (same voice ~0.75-0.85)" />
              </div>

              {(current.flags.length > 0 || current.duplicate_of) && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {current.flags.map((f) => (
                    <Badge key={f} color="amber" title={FLAG_INFO[f] || f}>{f}</Badge>
                  ))}
                  {current.duplicate_of && (
                    <span className="text-xs text-slate-500">
                      duplicate of clip <b>{current.duplicate_of}</b> (kept instead of this one)
                    </span>
                  )}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 rounded-md bg-white p-3 ring-1 ring-slate-200">
                <Button onMouseDown={noFocus} onClick={() => prev()} title="previous (←)">← Prev</Button>
                <div className="flex items-center">
                  <select className={`${inputBase} w-auto rounded-r-none`} value={rejectReason} onChange={(e) => setRejectReason(e.target.value)} title="reason used by R / bulk reject">
                    {config.reject_reasons.map((r) => <option key={r}>{r}</option>)}
                  </select>
                  <Button variant="danger" className="rounded-l-none" onMouseDown={noFocus} onClick={() => review('rejected')}>
                    Reject <kbd className="ml-1 border-rose-300 bg-rose-500 text-white">R</kbd>
                  </Button>
                </div>
                <Button variant="success" onMouseDown={noFocus} onClick={() => review('approved')}>
                  Approve <kbd className="ml-1 border-emerald-300 bg-emerald-500 text-white">A</kbd>
                </Button>
                <Button variant="primary" onMouseDown={noFocus} onClick={saveAndNext}>
                  Save &amp; next <kbd className="ml-1 border-indigo-300 bg-indigo-500 text-white">↵</kbd>
                </Button>
                {current.status !== 'pending' && (
                  <Button variant="ghost" onMouseDown={noFocus} onClick={() => review('pending', { advance: false })}>reset to pending</Button>
                )}
                <Button onMouseDown={noFocus} onClick={() => next()} title="next (→)" className="ml-auto">Next →</Button>
              </div>

              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
                <label className="flex items-center gap-1.5">
                  <input type="checkbox" checked={autoAdvance} onChange={(e) => setAutoAdvance(e.target.checked)} onMouseUp={(e) => e.currentTarget.blur()} />
                  next clip after approve / reject
                </label>
                <label className="flex items-center gap-1.5">
                  <input type="checkbox" checked={autoplay} onChange={(e) => setAutoplay(e.target.checked)} onMouseUp={(e) => e.currentTarget.blur()} />
                  autoplay
                </label>
                <label className="flex items-center gap-1.5">
                  speed
                  <select className="rounded border-0 bg-transparent text-xs" value={rate} onChange={(e) => setRate(Number(e.target.value))}>
                    {[0.75, 1, 1.25, 1.5, 1.75, 2].map((r) => <option key={r} value={r}>{r}×</option>)}
                  </select>
                </label>
              </div>
              <Shortcuts />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
