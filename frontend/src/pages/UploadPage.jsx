import { useEffect, useMemo, useRef, useState } from 'react'
import { api, uploadFile } from '../api.js'
import { navigate } from '../hooks.js'
import { Badge, Button, Card, Field, ProgressBar, fmtBytes, inputCls, useConfig, useToast } from '../components/ui.jsx'

let nextId = 1

function SpeakerFields({ value, onChange, speakers, compact = false }) {
  const config = useConfig()
  const existing = speakers.find((s) => s.name.toLowerCase() === (value.speaker_name || '').trim().toLowerCase())
  useEffect(() => {
    if (existing && existing.gender !== value.gender) onChange({ ...value, gender: existing.gender })
  }, [existing, value, onChange])
  const multi = !!value.multi_speaker
  const cls = compact ? 'grid grid-cols-[1.4fr_1fr_1fr_1.6fr] gap-2' : 'grid grid-cols-2 gap-3 md:grid-cols-4'
  return (
    <div>
    <div className={cls}>
      <Field label={compact ? null : 'Speaker name'} hint={!compact && (multi ? 'named after diarization' : existing ? `existing speaker (${existing.gender})` : 'new speakers are created automatically')}>
        <input className={inputCls} list="speaker-names" value={multi ? '' : value.speaker_name} disabled={multi}
          placeholder={multi ? 'after diarization' : 'e.g. ravi'} onChange={(e) => onChange({ ...value, speaker_name: e.target.value })} />
      </Field>
      <Field label={compact ? null : 'Gender'}>
        <select className={inputCls} value={multi ? '' : value.gender} disabled={!!existing || multi} onChange={(e) => onChange({ ...value, gender: e.target.value })}>
          <option value="">select…</option>
          {config.genders.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </Field>
      <Field label={compact ? null : 'Language'}>
        <select className={inputCls} value={value.language} onChange={(e) => onChange({ ...value, language: e.target.value })}>
          <option value="">select…</option>
          {config.languages.map((l) => (
            <option key={l.code} value={l.code}>{l.name} ({l.code})</option>
          ))}
        </select>
      </Field>
      <Field label={compact ? null : 'Notes (optional)'}>
        <input className={inputCls} value={value.notes} placeholder="mic, room, session…" onChange={(e) => onChange({ ...value, notes: e.target.value })} />
      </Field>
    </div>
    <label className={`flex items-center gap-2 text-slate-600 ${compact ? 'mt-1.5 text-xs' : 'mt-3 text-sm'}`}>
      <input type="checkbox" checked={multi} onChange={(e) => onChange({ ...value, multi_speaker: e.target.checked })} />
      Multiple speakers in this file
      {!compact && <span className="text-xs text-slate-400">the file is diarized first; you then listen to each detected voice and name it</span>}
    </label>
    </div>
  )
}

const metaComplete = (m) => m.language && (m.multi_speaker || (m.speaker_name.trim() && m.gender))

export default function UploadPage() {
  const config = useConfig()
  const toast = useToast()
  const [speakers, setSpeakers] = useState([])
  const [defaults, setDefaults] = useState({ speaker_name: '', gender: '', language: '', notes: '', multi_speaker: false })
  const [queue, setQueue] = useState([])
  const [autoProcess, setAutoProcess] = useState(true)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)

  useEffect(() => {
    api.speakers().then(setSpeakers).catch((e) => toast(e.message, 'error'))
  }, [toast])

  const accept = useMemo(() => config.allowed_extensions.map((e) => `.${e}`).join(','), [config])

  function addFiles(files) {
    const ok = []
    for (const f of files) {
      const ext = f.name.split('.').pop().toLowerCase()
      if (!config.allowed_extensions.includes(ext)) {
        toast(`${f.name}: unsupported type .${ext}`, 'error')
        continue
      }
      ok.push({ id: nextId++, file: f, meta: { ...defaults }, state: 'waiting', progress: 0, error: null, source: null })
    }
    setQueue((q) => [...q, ...ok])
  }

  const setItem = (id, patch) => setQueue((q) => q.map((it) => (it.id === id ? { ...it, ...patch } : it)))
  const waiting = queue.filter((it) => it.state === 'waiting' || it.state === 'error')
  const incomplete = waiting.filter((it) => !metaComplete(it.meta))

  async function uploadAll() {
    setBusy(true)
    for (const it of waiting) {
      if (!metaComplete(it.meta)) continue
      setItem(it.id, { state: 'uploading', progress: 0, error: null })
      try {
        const meta = it.meta.multi_speaker
          ? { language: it.meta.language, notes: it.meta.notes, multi_speaker: true }
          : { ...it.meta, multi_speaker: undefined }
        const src = await uploadFile(it.file, { ...meta, auto_process: autoProcess }, (p) => setItem(it.id, { progress: p }))
        setItem(it.id, { state: 'done', progress: 1, source: src })
      } catch (e) {
        setItem(it.id, { state: 'error', error: e.message })
      }
    }
    setBusy(false)
    api.speakers().then(setSpeakers).catch(() => {})
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      <datalist id="speaker-names">
        {speakers.map((s) => (
          <option key={s.id} value={s.name}>{s.gender}</option>
        ))}
      </datalist>

      <Card title="1 · Metadata for new files">
        <SpeakerFields value={defaults} onChange={setDefaults} speakers={speakers} />
        <p className="mt-3 text-xs text-slate-500">
          One upload = one speaker, unless you tick “Multiple speakers in this file”. These values pre-fill every file you add below; you can
          still change them per file. Originals are stored
          untouched under <code className="rounded bg-slate-100 px-1">data/raw/</code>.
        </p>
      </Card>

      <Card title="2 · Recordings">
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            addFiles(e.dataTransfer.files)
          }}
          onClick={() => fileInput.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors ${dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}
        >
          <div className="text-sm font-medium text-slate-700">Drop audio files here or click to choose</div>
          <div className="mt-1 text-xs text-slate-500">
            {config.allowed_extensions.join(', ')} · any length · up to {config.max_file_gb} GB each
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept={accept}
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files)
              e.target.value = ''
            }}
          />
        </div>

        {queue.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>{queue.length} file(s)</span>
              <div className="flex gap-2">
                <Button size="sm" variant="ghost" onClick={() => setQueue((q) => q.map((it) => (it.state === 'done' ? it : { ...it, meta: { ...defaults } })))}>
                  apply metadata to all
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setQueue((q) => q.filter((it) => it.state !== 'done'))}>
                  clear finished
                </Button>
              </div>
            </div>
            {queue.map((it) => (
              <div key={it.id} className="rounded-md border border-slate-200 p-3">
                <div className="mb-2 flex items-center gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-slate-800">{it.file.name}</div>
                    <div className="text-xs text-slate-500">{fmtBytes(it.file.size)}</div>
                  </div>
                  {it.state === 'waiting' && <Badge>waiting</Badge>}
                  {it.state === 'uploading' && <Badge color="blue">uploading {Math.round(it.progress * 100)}%</Badge>}
                  {it.state === 'done' && (
                    <button className="text-xs text-indigo-600 hover:underline" onClick={() => navigate('sources')}>
                      <Badge color="green">uploaded as #{it.source?.id}{it.source?.job ? ' · processing queued' : ''}</Badge>
                    </button>
                  )}
                  {it.state === 'error' && <Badge color="red">failed</Badge>}
                  {it.state !== 'uploading' && it.state !== 'done' && (
                    <Button size="sm" variant="ghost" onClick={() => setQueue((q) => q.filter((x) => x.id !== it.id))}>
                      remove
                    </Button>
                  )}
                </div>
                {it.state !== 'done' && (
                  <SpeakerFields compact value={it.meta} speakers={speakers} onChange={(meta) => setItem(it.id, { meta })} />
                )}
                {it.state === 'uploading' && <ProgressBar className="mt-2" value={it.progress * 100} />}
                {it.error && <div className="mt-2 text-xs text-rose-600">{it.error}</div>}
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="flex items-center justify-between rounded-lg bg-white px-4 py-3 shadow-sm ring-1 ring-slate-200">
        <label className="flex items-center gap-2 text-sm text-slate-700">
          <input type="checkbox" checked={autoProcess} onChange={(e) => setAutoProcess(e.target.checked)} />
          Start processing automatically after upload
        </label>
        <div className="flex items-center gap-3">
          {incomplete.length > 0 && <span className="text-xs text-amber-700">{incomplete.length} file(s) missing speaker / gender / language</span>}
          <Button variant="primary" disabled={busy || !waiting.length || incomplete.length === waiting.length} onClick={uploadAll}>
            {busy ? 'Uploading…' : `Upload ${waiting.length - incomplete.length || ''} file(s)`}
          </Button>
        </div>
      </div>
    </div>
  )
}
