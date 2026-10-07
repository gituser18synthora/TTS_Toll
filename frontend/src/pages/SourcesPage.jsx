import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { navigate, usePoll } from '../hooks.js'
import {
  Badge, Button, Card, ConfirmDialog, Field, Modal, ProgressBar, StatusBadge, fmtDuration, fmtHours, fmtTime, inputCls,
  languageName, useConfig, useToast,
} from '../components/ui.jsx'

const ACTIVE = (s) => s.job && (s.job.status === 'queued' || s.job.status === 'running')

export function JobLogModal({ jobId, title, onClose }) {
  const { data: job } = usePoll(() => api.job(jobId), (j) => (j && (j.status === 'running' || j.status === 'queued') ? 1500 : 0), [jobId])
  return (
    <Modal title={title || `Job #${jobId}`} onClose={onClose} wide>
      {job ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <StatusBadge status={job.status} /> {job.type} · {Math.round(job.progress)}% · {job.message}
          </div>
          {job.error && <div className="rounded bg-rose-50 p-2 text-sm text-rose-700">{job.error}</div>}
          <pre className="max-h-[60vh] overflow-auto rounded bg-slate-900 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-slate-100">
            {job.log || '(empty)'}
          </pre>
        </div>
      ) : (
        <div className="text-sm text-slate-500">loading…</div>
      )}
    </Modal>
  )
}

function EditSourceModal({ source, onClose, onSaved }) {
  const config = useConfig()
  const toast = useToast()
  const multi = source.multi_speaker
  const [form, setForm] = useState({ speaker_name: source.speaker_name, gender: source.gender, language: source.language, notes: source.notes || '' })
  const [busy, setBusy] = useState(false)
  async function save() {
    setBusy(true)
    try {
      await api.updateSource(source.id, multi ? { language: form.language, notes: form.notes } : form)
      toast('Metadata updated', 'success')
      onSaved()
      onClose()
    } catch (e) {
      toast(e.message, 'error')
    } finally {
      setBusy(false)
    }
  }
  return (
    <Modal
      title={`Edit #${source.id} · ${source.original_filename}`}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={busy} onClick={save}>Save</Button>
        </>
      }
    >
      <div className="grid grid-cols-2 gap-3">
        {!multi && <>
        <Field label="Speaker name" hint="Changing it moves all clips of this file to that speaker">
          <input className={inputCls} value={form.speaker_name} onChange={(e) => setForm({ ...form, speaker_name: e.target.value })} />
        </Field>
        <Field label="Gender (for a new speaker)">
          <select className={inputCls} value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}>
            {config.genders.map((g) => <option key={g}>{g}</option>)}
          </select>
        </Field>
        </>}
        <Field label="Language" hint="Re-run processing afterwards to re-transcribe with the new language">
          <select className={inputCls} value={form.language} onChange={(e) => setForm({ ...form, language: e.target.value })}>
            {config.languages.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
          </select>
        </Field>
        <Field label="Notes">
          <input className={inputCls} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
        </Field>
      </div>
    </Modal>
  )
}

function SpeakerAssignModal({ source, onClose, onSaved }) {
  const config = useConfig()
  const toast = useToast()
  const [view, setView] = useState(null)
  const [known, setKnown] = useState([])
  const [rows, setRows] = useState({})
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.speakers().then(setKnown).catch(() => {})
    api.sourceSpeakers(source.id).then((v) => {
      setView(v)
      setRows(Object.fromEntries(v.labels.map((l) => [l.label, { ignore: l.ignore, name: l.speaker_name || '', gender: l.gender || '' }])))
    }).catch((e) => toast(e.message, 'error'))
  }, [source.id, toast])

  const setRow = (label, patch) => setRows((r) => ({ ...r, [label]: { ...r[label], ...patch } }))
  const existing = (name) => known.find((k) => k.name.toLowerCase() === name.trim().toLowerCase())
  const ready = view && view.labels.every((l) => rows[l.label]?.ignore || (rows[l.label]?.name.trim() && (rows[l.label]?.gender || existing(rows[l.label].name))))

  async function save() {
    setBusy(true)
    try {
      const assignments = view.labels.map((l) => {
        const r = rows[l.label]
        return r.ignore ? { label: l.label, ignore: true } : { label: l.label, name: r.name.trim(), gender: existing(r.name)?.gender || r.gender }
      })
      await api.assignSpeakers(source.id, assignments)
      toast(view.clips_created ? 'Speakers renamed' : 'Speakers saved, processing continues', 'success')
      onSaved()
      onClose()
    } catch (e) {
      toast(e.message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal title={`Speakers in ${source.original_filename}`} onClose={onClose} wide
      footer={<><Button onClick={onClose}>Cancel</Button><Button variant="primary" disabled={!ready || busy} onClick={save}>Save speakers</Button></>}>
      {!view ? (
        <div className="text-sm text-slate-500">loading…</div>
      ) : (
        <div className="space-y-3">
          <p className="text-xs text-slate-500">
            {view.labels.length} voice(s) found by {view.engine}
            {view.overlap_seconds > 0 && ` · ${view.overlap_seconds} s of overlapping speech (those clips are flagged or rejected)`}.
            Listen to the samples, then name each voice (the same name for two labels merges them) or ignore it.
            {view.clips_created && ' Clips already exist: you can rename, but changing what is ignored needs Reprocess.'}
          </p>
          {view.labels.map((l) => {
            const r = rows[l.label] || {}
            const ex = r.name ? existing(r.name) : null
            return (
              <div key={l.label} className={`rounded-md border p-3 ${r.ignore ? 'border-slate-200 bg-slate-50' : 'border-slate-200'}`}>
                <div className="mb-2 flex items-center gap-3 text-sm">
                  <b className="text-slate-800">{l.label}</b>
                  <span className="text-xs text-slate-500">{(l.seconds / 60).toFixed(1)} min · {l.turns} turns</span>
                </div>
                <div className="mb-2 flex flex-wrap gap-2">
                  {l.samples.map((url) => <audio key={url} controls preload="none" src={url} className="h-8 w-56" />)}
                  {!l.samples.length && <span className="text-xs text-slate-400">no sample long enough</span>}
                </div>
                <div className="grid grid-cols-[1fr_10rem_auto] items-center gap-2">
                  <input className={inputCls} list="speaker-names-assign" placeholder="speaker name" value={r.name} disabled={r.ignore}
                    onChange={(e) => setRow(l.label, { name: e.target.value })} />
                  <select className={inputCls} value={ex ? ex.gender : r.gender} disabled={r.ignore || !!ex}
                    onChange={(e) => setRow(l.label, { gender: e.target.value })}>
                    <option value="">gender…</option>
                    {config.genders.map((g) => <option key={g}>{g}</option>)}
                  </select>
                  <label className="flex items-center gap-1.5 text-xs text-slate-600">
                    <input type="checkbox" checked={r.ignore} disabled={view.clips_created}
                      onChange={(e) => setRow(l.label, { ignore: e.target.checked })} />
                    ignore
                  </label>
                </div>
              </div>
            )
          })}
          <datalist id="speaker-names-assign">{known.map((k) => <option key={k.id} value={k.name}>{k.gender}</option>)}</datalist>
        </div>
      )}
    </Modal>
  )
}

export default function SourcesPage() {
  const config = useConfig()
  const toast = useToast()
  const [logJob, setLogJob] = useState(null)
  const [editing, setEditing] = useState(null)
  const [confirm, setConfirm] = useState(null)
  const [assigning, setAssigning] = useState(null)
  const { data: sources, error, reload } = usePoll(() => api.sources(), (d) => (d?.some(ACTIVE) ? 2000 : 10000))

  async function act(fn, okMsg) {
    try {
      await fn()
      if (okMsg) toast(okMsg, 'success')
    } catch (e) {
      toast(e.message, 'error')
    }
    reload()
  }

  const totals = (sources || []).reduce(
    (t, s) => ({ dur: t.dur + (s.duration || 0), approved: t.approved + s.counts.approved_hours, clips: t.clips + s.counts.total }),
    { dur: 0, approved: 0, clips: 0 },
  )

  return (
    <div className="space-y-4 p-4">
      <Card
        title={`Source recordings${sources ? ` (${sources.length})` : ''}`}
        actions={
          <>
            {sources && (
              <span className="text-xs text-slate-500">
                raw {fmtHours(totals.dur / 3600)} · {totals.clips} clips · approved {fmtHours(totals.approved)}
              </span>
            )}
            {sources?.some((x) => x.steps_done.includes('clips')) && (
              <Button
                size="sm"
                onClick={() =>
                  setConfirm({
                    title: 'Re-transcribe every file?',
                    message: (
                      <>
                        Runs both ASR systems again on all existing clips (no re-segmentation), e.g. after moving to a GPU with
                        large-v3. Approved and rejected clips keep their status, and transcripts you edited are kept. Only
                        pending, never-edited transcripts get the new text.
                      </>
                    ),
                    label: 'Re-transcribe all',
                    danger: false,
                    run: () => act(async () => {
                      const r = await api.retranscribeAll()
                      toast(`${r.queued.length} file(s) queued${r.skipped.length ? `, ${r.skipped.length} skipped (busy or not processed)` : ''}`, 'success')
                    }),
                  })
                }
              >
                Re-transcribe all
              </Button>
            )}
            <Button size="sm" variant="ghost" title="re-check every speaker's clips against their voice (speaker_mismatch)"
              onClick={() => act(async () => { const j = await api.speakerCheckAll(); toast(`Speaker check queued (job #${j.id})`, 'success') })}>
              Re-check speakers
            </Button>
            <Button size="sm" variant="ghost" title="find exact / near-identical duplicate clips across all files"
              onClick={() => act(async () => { const j = await api.dedupAll(); toast(`Duplicate search queued (job #${j.id})`, 'success') })}>
              Find duplicates
            </Button>
            <Button size="sm" variant="primary" onClick={() => navigate('upload')}>Upload</Button>
          </>
        }
      >
        {error && <div className="mb-3 rounded bg-rose-50 p-2 text-sm text-rose-700">{error}</div>}
        {sources && !sources.length && (
          <div className="py-10 text-center text-sm text-slate-500">
            No recordings yet. <button className="text-indigo-600 hover:underline" onClick={() => navigate('upload')}>Upload some</button>.
          </div>
        )}
        {sources?.length > 0 && (
          <div className="-mx-4 -mb-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-slate-500">
                <tr>
                  <th className="px-4 py-2 font-medium">#</th>
                  <th className="px-2 py-2 font-medium">File</th>
                  <th className="px-2 py-2 font-medium">Speaker</th>
                  <th className="px-2 py-2 font-medium">Lang</th>
                  <th className="px-2 py-2 text-right font-medium">Length</th>
                  <th className="w-64 px-2 py-2 font-medium">Status</th>
                  <th className="px-2 py-2 font-medium">Clips</th>
                  <th className="px-4 py-2 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {sources.map((s) => {
                  const active = ACTIVE(s)
                  const c = s.counts
                  return (
                    <tr key={s.id} className="align-top hover:bg-slate-50/60">
                      <td className="px-4 py-2.5 text-slate-400 tabular-nums">{s.id}</td>
                      <td className="max-w-xs px-2 py-2.5">
                        <div className="truncate font-medium text-slate-800" title={s.original_filename}>{s.original_filename}</div>
                        <div className="text-xs text-slate-400">
                          {fmtTime(s.created_at)}
                          {s.enhanced && <span className="ml-1.5 rounded bg-sky-50 px-1 text-sky-700 ring-1 ring-sky-200">denoised</span>}
                          {s.notes ? ` · ${s.notes}` : ''}
                        </div>
                      </td>
                      <td className="px-2 py-2.5">
                        {s.speaker_name} {s.gender && <span className="text-xs text-slate-400">{s.gender[0].toUpperCase()}</span>}
                        {s.multi_speaker && <span className="ml-1 rounded bg-violet-50 px-1 text-[11px] text-violet-700 ring-1 ring-violet-200">multi</span>}
                      </td>
                      <td className="px-2 py-2.5">{languageName(config, s.language)}</td>
                      <td className="px-2 py-2.5 text-right tabular-nums">{fmtDuration(s.duration)}</td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-2">
                          <StatusBadge status={active && s.job.status === 'queued' ? 'queued' : s.status === 'needs_speakers' ? 'needs_speakers' : s.status} />
                          {s.status === 'needs_speakers' && <span className="text-xs text-amber-700">name the detected voices</span>}
                          {active && s.job.status === 'running' && <span className="truncate text-xs text-slate-500">{s.job.message}</span>}
                        </div>
                        {active && <ProgressBar className="mt-1.5" value={s.job.progress} />}
                        {s.status === 'failed' && s.error && <div className="mt-1 line-clamp-2 text-xs text-rose-600" title={s.error}>{s.error}</div>}
                      </td>
                      <td className="px-2 py-2.5 text-xs">
                        {c.total > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            <Badge color="green" title="approved">{c.approved} ✓</Badge>
                            <Badge title="pending">{c.pending} pending</Badge>
                            {c.rejected > 0 && <Badge color="red" title="rejected">{c.rejected} ✕</Badge>}
                            {c.needs_review > 0 && <Badge color="amber" title="pending clips flagged for review">{c.needs_review} flagged</Badge>}
                          </div>
                        ) : (
                          <span className="text-slate-400">–</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex justify-end gap-1">
                          {s.status === 'needs_speakers' && (
                            <Button size="sm" variant="primary" onClick={() => setAssigning(s)}>Assign speakers</Button>
                          )}
                          {c.total > 0 && (
                            <Button size="sm" variant="primary" onClick={() => navigate('review', { source_file_id: s.id, status: 'pending' })}>
                              Review
                            </Button>
                          )}
                          {s.multi_speaker && s.status !== 'needs_speakers' && s.steps_done.includes('diarize') && !active && (
                            <Button size="sm" variant="ghost" onClick={() => setAssigning(s)}>Speakers</Button>
                          )}
                          {s.job && (
                            <Button size="sm" variant="ghost" onClick={() => setLogJob(s.job.id)}>Log</Button>
                          )}
                          {active && (
                            <Button size="sm" variant="ghost" onClick={() => act(() => api.cancelJob(s.job.id), 'Cancel requested')}>Cancel</Button>
                          )}
                          {!active && s.status !== 'done' && (
                            <Button size="sm" onClick={() => act(() => api.processSource(s.id), 'Queued (resumes from the last completed step)')}>
                              {s.status === 'failed' ? 'Retry' : 'Process'}
                            </Button>
                          )}
                          {!active && s.steps_done.includes('clips') && (
                            <Button
                              size="sm"
                              variant="ghost"
                              title="run both ASRs again on the existing clips; keeps reviews and edits"
                              onClick={() => act(() => api.retranscribeSource(s.id), 'Re-transcription queued')}
                            >
                              Re-transcribe
                            </Button>
                          )}
                          {!active && (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() =>
                                setConfirm({
                                  title: `Reprocess ${s.original_filename}?`,
                                  message: (
                                    <>
                                      All <b>{c.total}</b> clips of this file
                                      {c.approved + c.rejected > 0 && <> — including <b>{c.approved + c.rejected}</b> already reviewed</>} — and its derived audio
                                      will be deleted and regenerated from the original upload with the current config. The original is kept.
                                    </>
                                  ),
                                  label: 'Delete clips & reprocess',
                                  danger: true,
                                  run: () => act(() => api.reprocessSource(s.id), 'Reprocessing queued'),
                                })
                              }
                            >
                              Reprocess
                            </Button>
                          )}
                          {!active && <Button size="sm" variant="ghost" onClick={() => setEditing(s)}>Edit</Button>}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {logJob && <JobLogModal jobId={logJob} onClose={() => setLogJob(null)} />}
      {editing && <EditSourceModal source={editing} onClose={() => setEditing(null)} onSaved={reload} />}
      {assigning && <SpeakerAssignModal source={assigning} onClose={() => setAssigning(null)} onSaved={reload} />}
      {confirm && (
        <ConfirmDialog
          title={confirm.title}
          message={confirm.message}
          danger={confirm.danger}
          confirmLabel={confirm.label}
          onCancel={() => setConfirm(null)}
          onConfirm={async () => {
            await confirm.run()
            setConfirm(null)
          }}
        />
      )}
    </div>
  )
}
