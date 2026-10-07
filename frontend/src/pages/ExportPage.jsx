import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { usePoll } from '../hooks.js'
import { Badge, Button, Card, Field, ProgressBar, StatusBadge, fmtBytes, fmtHours, fmtTime, inputCls, languageName, useConfig, useToast } from '../components/ui.jsx'
import { JobLogModal } from './SourcesPage.jsx'

const today = () => new Date().toISOString().slice(0, 10).replaceAll('-', '')

function toggle(list, v) {
  return list.includes(v) ? list.filter((x) => x !== v) : [...list, v]
}

export default function ExportPage() {
  const config = useConfig()
  const toast = useToast()
  const [speakers, setSpeakers] = useState([])
  const [form, setForm] = useState({ name: `orpheus_${today()}`, speaker_ids: [], languages: [], csv: true, hf: false, script: 'devanagari', verify_audio: true, make_zip: true })
  const [preview, setPreview] = useState(null)
  const [busy, setBusy] = useState(false)
  const [logJob, setLogJob] = useState(null)
  const exports = usePoll(() => api.exports(), (d) => (d?.some((e) => e.status === 'queued' || e.status === 'running') ? 1500 : 15000))

  useEffect(() => {
    api.speakers().then(setSpeakers).catch((e) => toast(e.message, 'error'))
  }, [toast])

  useEffect(() => {
    const t = setTimeout(() => {
      api.exportPreview({ speaker_ids: form.speaker_ids, languages: form.languages }).then(setPreview).catch(() => {})
    }, 200)
    return () => clearTimeout(t)
  }, [form.speaker_ids, form.languages])

  async function start() {
    setBusy(true)
    try {
      await api.createExport({
        name: form.name,
        speaker_ids: form.speaker_ids,
        languages: form.languages,
        formats: ['jsonl', ...(form.csv ? ['csv'] : []), ...(form.hf ? ['hf'] : [])],
        script: form.script,
        verify_audio: form.verify_audio,
        make_zip: form.make_zip,
      })
      toast('Export queued', 'success')
      exports.reload()
    } catch (e) {
      toast(e.message, 'error')
    } finally {
      setBusy(false)
    }
  }

  async function rerun(id) {
    try {
      await api.rerunExport(id)
      exports.reload()
    } catch (e) {
      toast(e.message, 'error')
    }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Card title="New export (approved clips only)">
          <div className="space-y-4">
            <Field label="Export name" hint="Re-running an export with the same name rebuilds data/exports/<name>/ and <name>.zip">
              <input className={inputCls} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </Field>
            <div>
              <div className="mb-1 text-xs font-medium text-slate-600">Speakers {form.speaker_ids.length === 0 && <span className="font-normal text-slate-400">(none selected = all)</span>}</div>
              <div className="flex flex-wrap gap-2">
                {speakers.map((s) => (
                  <label key={s.id} className={`flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-sm ring-1 ${form.speaker_ids.includes(s.id) ? 'bg-indigo-50 ring-indigo-300' : 'ring-slate-200'}`}>
                    <input type="checkbox" checked={form.speaker_ids.includes(s.id)} onChange={() => setForm({ ...form, speaker_ids: toggle(form.speaker_ids, s.id) })} />
                    {s.name} <span className="text-xs text-slate-400">{fmtHours(s.approved_hours)}</span>
                  </label>
                ))}
                {!speakers.length && <span className="text-sm text-slate-400">no speakers yet</span>}
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-slate-600">Languages {form.languages.length === 0 && <span className="font-normal text-slate-400">(none selected = all)</span>}</div>
              <div className="flex flex-wrap gap-2">
                {config.languages.map((l) => (
                  <label key={l.code} className={`flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-sm ring-1 ${form.languages.includes(l.code) ? 'bg-indigo-50 ring-indigo-300' : 'ring-slate-200'}`}>
                    <input type="checkbox" checked={form.languages.includes(l.code)} onChange={() => setForm({ ...form, languages: toggle(form.languages, l.code) })} />
                    {l.name}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-slate-600">
                Hindi / Hinglish script <span className="font-normal text-slate-400">(other languages always export as stored)</span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex rounded-md bg-slate-100 p-0.5 text-sm ring-1 ring-slate-200">
                  {[['devanagari', 'Devanagari'], ['roman', 'Roman'], ['both', 'Both']].map(([v, label]) => (
                    <button key={v} type="button" onClick={() => setForm({ ...form, script: v })}
                      className={`rounded px-3 py-1 ${form.script === v ? 'bg-white font-medium text-slate-800 shadow-sm' : 'text-slate-500'}`}>
                      {label}
                    </button>
                  ))}
                </div>
                <span className="text-xs text-slate-500">
                  {form.script === 'devanagari' && 'as stored: Devanagari, English words in Latin script'}
                  {form.script === 'roman' && `romanized (${config.roman_scheme} spelling, e.g. "aaj mausam achha hai")`}
                  {form.script === 'both' && 'one row per script, both pointing at the same wav'}
                </span>
              </div>
            </div>
            <div className="grid gap-2 text-sm sm:grid-cols-2">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked disabled /> train.jsonl (Orpheus / Unsloth)
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.csv} onChange={(e) => setForm({ ...form, csv: e.target.checked })} /> metadata.csv (LJSpeech-style)
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.hf} onChange={(e) => setForm({ ...form, hf: e.target.checked })} /> HuggingFace datasets (save_to_disk)
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.verify_audio} onChange={(e) => setForm({ ...form, verify_audio: e.target.checked })} /> verify / fix 24 kHz mono 16-bit
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.make_zip} onChange={(e) => setForm({ ...form, make_zip: e.target.checked })} /> build .zip (audio + manifests + SHA256SUMS)
              </label>
            </div>
            <div className="flex justify-end">
              <Button variant="primary" disabled={busy || !preview?.clips || !form.name.trim()} onClick={start}>
                Export {preview ? `${preview.clips} clips · ${fmtHours(preview.hours)}` : ''}
                {preview && form.script === 'both' && preview.romanizable_clips > 0 ? ` (${preview.clips + preview.romanizable_clips} rows)` : ''}
              </Button>
            </div>
          </div>
        </Card>

        <Card title="Preview">
          {!preview ? (
            <div className="text-sm text-slate-400">…</div>
          ) : !preview.clips ? (
            <div className="text-sm text-slate-500">No approved clips match. Approve clips on the Review page first.</div>
          ) : (
            <div className="space-y-2">
              <div className="text-2xl font-semibold text-slate-800">
                {fmtHours(preview.hours)} <span className="text-sm font-normal text-slate-500">in {preview.clips} clips</span>
              </div>
              {preview.empty_text > 0 && <div className="text-xs text-amber-700">{preview.empty_text} approved clip(s) have an empty transcript and will be skipped</div>}
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-slate-500">
                  <tr><th className="py-1 font-medium">speaker</th><th className="py-1 font-medium">language</th><th className="py-1 text-right font-medium">clips</th><th className="py-1 text-right font-medium">audio</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {preview.groups.map((g) => (
                    <tr key={`${g.speaker}/${g.language}`}>
                      <td className="py-1">{g.speaker}</td>
                      <td className="py-1">{languageName(config, g.language)}</td>
                      <td className="py-1 text-right tabular-nums">{g.clips}</td>
                      <td className="py-1 text-right tabular-nums">{fmtHours(g.hours)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card title="Exports">
        {!exports.data?.length ? (
          <div className="text-sm text-slate-500">No exports yet.</div>
        ) : (
          <div className="-mx-4 -mb-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-slate-500">
                <tr>
                  <th className="px-4 py-2 font-medium">name</th>
                  <th className="px-2 py-2 font-medium">created</th>
                  <th className="w-56 px-2 py-2 font-medium">status</th>
                  <th className="px-2 py-2 text-right font-medium">clips</th>
                  <th className="px-2 py-2 text-right font-medium">audio</th>
                  <th className="px-2 py-2 font-medium">location</th>
                  <th className="px-4 py-2 text-right font-medium">actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {exports.data.map((e) => {
                  const active = e.status === 'queued' || e.status === 'running'
                  return (
                    <tr key={e.id} className="align-top">
                      <td className="px-4 py-2 font-medium text-slate-800">{e.name}</td>
                      <td className="px-2 py-2 text-xs text-slate-500">{fmtTime(e.created_at)}</td>
                      <td className="px-2 py-2">
                        <div className="flex items-center gap-2">
                          <StatusBadge status={e.status} />
                          {active && e.job && <span className="truncate text-xs text-slate-500">{e.job.message}</span>}
                        </div>
                        {active && e.job && <ProgressBar className="mt-1.5" value={e.job.progress} />}
                        {e.error && <div className="mt-1 text-xs text-rose-600">{e.error}</div>}
                        {e.stats?.skipped > 0 && <Badge color="amber" className="mt-1">{e.stats.skipped} skipped</Badge>}
                        {e.params?.script && e.params.script !== 'devanagari' && <Badge color="indigo" className="mt-1 ml-1">{e.params.script}</Badge>}
                        {e.params?.formats?.includes('hf') && <Badge className="mt-1 ml-1">+ HF dataset</Badge>}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">{e.status === 'done' ? e.clip_count : '–'}</td>
                      <td className="px-2 py-2 text-right tabular-nums">{e.status === 'done' ? fmtHours(e.total_duration / 3600) : '–'}</td>
                      <td className="px-2 py-2 font-mono text-xs text-slate-500">
                        {e.out_dir && <div>data/{e.out_dir}/</div>}
                        {e.zip_path && <div>data/{e.zip_path} · {fmtBytes(e.zip_size)}</div>}
                      </td>
                      <td className="px-4 py-2">
                        <div className="flex justify-end gap-1">
                          {e.zip_path && e.status === 'done' && (
                            <a href={`/api/exports/${e.id}/download`} className="inline-flex items-center rounded-md bg-indigo-600 px-2 py-1 text-xs font-medium text-white hover:bg-indigo-700">
                              Download zip
                            </a>
                          )}
                          {e.job && <Button size="sm" variant="ghost" onClick={() => setLogJob(e.job.id)}>Log</Button>}
                          {!active && <Button size="sm" variant="ghost" onClick={() => rerun(e.id)} title="rebuild from the currently approved clips">Rebuild</Button>}
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
    </div>
  )
}
