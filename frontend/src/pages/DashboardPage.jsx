import { api } from '../api.js'
import { navigate, usePoll } from '../hooks.js'
import { ChartCard, Columns, DataTable, HBars, Legend, StackedBars } from '../components/charts.jsx'
import { Badge, ProgressBar, StatusBadge, fmtHours, languageName, useConfig } from '../components/ui.jsx'

const STATUS_KEYS = [
  { key: 'approved', label: 'Approved', icon: '✓', color: 'var(--viz-good)' },
  { key: 'pending', label: 'Pending', icon: '○', color: 'var(--viz-neutral)' },
  { key: 'rejected', label: 'Rejected', icon: '✕', color: 'var(--viz-critical)' },
]

const FLAG_LABELS = {
  asr_disagreement: 'ASR outputs disagree', low_asr_confidence: 'low ASR confidence', asr_no_speech: 'ASR: no speech',
  asr_repetition: 'ASR repetition', empty_transcript: 'empty transcript', sparse_transcript: 'sparse transcript',
  numbers_expanded: 'numbers spelled out', digits_remaining: 'digits left', chars_removed: 'characters removed',
  annotation_removed: 'non-speech tag', borderline_quality: 'borderline quality', music: 'music', babble: 'background talk',
  forced_cut: 'forced cut', very_quiet: 'very quiet', no_asr: 'no ASR', wrong_script: 'wrong script',
  speaker_mismatch: 'voice mismatch', duplicate: 'duplicate', overlap: 'overlapping speech',
}

const hrs = (h) => fmtHours(h)

function StatTile({ label, value, sub, hero = false, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      className="viz-root h-full w-full rounded-lg bg-[var(--viz-surface)] p-4 text-left shadow-sm ring-1 ring-[var(--viz-border)] enabled:hover:ring-indigo-300"
    >
      <div className="text-xs text-[var(--viz-text-2)]">{label}</div>
      <div className={`mt-1 font-semibold text-[var(--viz-text-1)] ${hero ? 'text-5xl leading-tight' : 'text-2xl'}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-[var(--viz-text-3)]">{sub}</div>}
    </button>
  )
}

function breakdownRows(obj, labelOf) {
  return Object.entries(obj || {})
    .map(([k, v]) => ({ label: labelOf(k), key: k, values: { approved: v.approved, pending: v.pending, rejected: v.rejected }, clips: v.clips }))
    .sort((a, b) => b.values.approved + b.values.pending - (a.values.approved + a.values.pending))
}

function EnginesCard({ health }) {
  const engines = health?.worker?.engines || {}
  const caps = health?.capabilities || {}
  const rows = [
    ['Denoise', engines.denoise],
    ['Quality', engines.quality],
    ['Music / babble tagger', engines.tagger],
    ['Primary ASR', engines.asr_primary],
    ['Secondary ASR', engines.asr_secondary],
  ]
  return (
    <section className="viz-root rounded-lg bg-[var(--viz-surface)] p-4 shadow-sm ring-1 ring-[var(--viz-border)]">
      <h2 className="text-sm font-semibold text-[var(--viz-text-1)]">Models in use</h2>
      <p className="mt-0.5 mb-3 text-xs text-[var(--viz-text-3)]">as reported by the worker once each step has run</p>
      <dl className="space-y-1.5 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex gap-3">
            <dt className="w-36 shrink-0 text-[var(--viz-text-2)]">{k}</dt>
            <dd className={`min-w-0 break-words ${v ? 'text-[var(--viz-text-1)]' : 'text-[var(--viz-text-3)] italic'}`}>{v || 'not used yet'}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {[['hf_token', 'HF token'], ['deepfilternet', 'DeepFilterNet'], ['dnsmos', 'DNSMOS'], ['silero_vad', 'Silero VAD'], ['faster_whisper', 'faster-whisper'], ['datasets', 'datasets']].map(([k, label]) => (
          <Badge key={k} color={caps[k] ? 'green' : 'slate'}>
            {caps[k] ? '✓' : '–'} {label}
          </Badge>
        ))}
      </div>
    </section>
  )
}

export default function DashboardPage() {
  const config = useConfig()
  const { data: s } = usePoll(() => api.stats(), (d) => (d && d.jobs.running + d.jobs.queued > 0 ? 3000 : 15000))
  const { data: health } = usePoll(() => api.health(), 10000)
  const { data: jobs } = usePoll(() => api.jobs({ status: 'queued,running', limit: 10 }), (d) => (d?.length ? 2000 : 10000))
  if (!s) return <div className="p-8 text-sm text-slate-500">loading…</div>

  const approvedH = s.by_status.approved?.hours || 0
  const pendingH = s.by_status.pending?.hours || 0
  const rejectedH = s.by_status.rejected?.hours || 0
  const today = new Date().toISOString().slice(0, 10)
  const todayRow = s.review_per_day.find((d) => d.day === today)

  const speakers = breakdownRows(s.per_speaker, (k) => `${k}${s.genders?.[k] ? ` (${s.genders[k][0].toUpperCase()})` : ''}`)
  const languages = breakdownRows(s.per_language, (k) => languageName(config, k))
  const qBins = s.quality_hist.bins
  const qData = qBins.slice(0, -1).map((b, i) => {
    const n = ['approved', 'pending', 'rejected'].reduce((t, k) => t + s.quality_hist.counts[k][i], 0)
    return { key: b, label: `${b.toFixed(2)}–${qBins[i + 1].toFixed(2)}`, edge: i % 4 === 0 ? b.toFixed(0) : '', value: n }
  })
  const qTotal = qData.reduce((t, d) => t + d.value, 0)
  const thresholds = config.quality
    ? [
        { at: (config.quality.reject_below - 1) / 0.25, label: `reject < ${config.quality.reject_below}`, side: 'left' },
        { at: (config.quality.review_below - 1) / 0.25, label: `review < ${config.quality.review_below}`, side: 'right' },
      ]
    : []
  const days = s.review_per_day.map((d) => ({
    key: d.day,
    label: d.day,
    short: d.day.slice(5),
    value: d.approved_hours * 60,
    tip: (
      <div>
        <div className="mb-1 font-medium text-[var(--viz-text-1)]">{d.day}</div>
        <div className="flex justify-between gap-4"><span className="text-[var(--viz-text-2)]">approved</span><b>{d.approved} clips · {(d.approved_hours * 60).toFixed(1)} min</b></div>
        <div className="flex justify-between gap-4"><span className="text-[var(--viz-text-2)]">rejected</span><b>{d.rejected} clips</b></div>
      </div>
    ),
  }))
  const flags = Object.entries(s.flags).map(([k, v]) => ({ label: FLAG_LABELS[k] || k, title: k, value: v }))
  const rejects = Object.entries(s.reject_reasons).map(([k, v]) => ({ label: k || 'unspecified', value: v })).sort((a, b) => b.value - a.value)
  const statusTable = (rows) => (
    <DataTable
      columns={[{ key: 'label', label: '' }, ...STATUS_KEYS.map((k) => ({ key: k.key, label: `${k.icon} ${k.label}`, num: true, fmt: hrs })), { key: 'clips', label: 'clips', num: true }]}
      rows={rows.map((r) => ({ label: r.label, ...r.values, clips: r.clips }))}
    />
  )

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <div className="col-span-2">
          <StatTile hero label="Approved, ready to export" value={hrs(approvedH)}
            sub={`${s.by_status.approved?.clips || 0} clips${todayRow ? ` · +${(todayRow.approved_hours * 60).toFixed(1)} min today` : ''}`}
            onClick={() => navigate('export')} />
        </div>
        <StatTile label="Pending review" value={hrs(pendingH)} sub={`${s.by_status.pending?.clips || 0} clips`} onClick={() => navigate('review', { status: 'pending' })} />
        <StatTile label="Flagged for a listen" value={s.needs_review.toLocaleString()} sub="pending clips with flags" onClick={() => navigate('review', { status: 'pending', needs_review: 'true' })} />
        <StatTile label="Mean quality (DNSMOS)" value={s.avg_quality != null ? s.avg_quality.toFixed(2) : '–'} sub={s.asr_agreement.mean_wer != null ? `mean ASR disagreement ${(s.asr_agreement.mean_wer * 100).toFixed(0)}% WER` : 'scale 1–5'} />
        <StatTile label="Raw audio uploaded" value={hrs(s.raw_hours)} sub={`rejected ${hrs(rejectedH)} · jobs ${s.jobs.running} running, ${s.jobs.queued} queued`} onClick={() => navigate('sources')} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Hours per speaker" subtitle="clip audio by review state" empty={!speakers.length && 'No clips yet'}
          legend={<Legend keys={STATUS_KEYS} />} table={statusTable(speakers)}>
          <StackedBars rows={speakers} keys={STATUS_KEYS} fmt={hrs} />
        </ChartCard>
        <ChartCard title="Hours per language" subtitle="clip audio by review state" empty={!languages.length && 'No clips yet'}
          legend={<Legend keys={STATUS_KEYS} />} table={statusTable(languages)}>
          <StackedBars rows={languages} keys={STATUS_KEYS} fmt={hrs} />
        </ChartCard>

        <ChartCard title="Clip quality" subtitle="DNSMOS OVRL distribution, all clips (1 = bad, 5 = excellent)" empty={!qTotal && 'No quality scores yet'}
          table={<DataTable columns={[{ key: 'label', label: 'OVRL' }, { key: 'value', label: 'clips', num: true }]} rows={qData.filter((d) => d.value)} />}>
          <Columns data={qData} color="var(--viz-series-1)" fmt={(v) => Math.round(v).toLocaleString()} markers={thresholds} tipLabel="OVRL " />
        </ChartCard>
        <ChartCard title="Review progress" subtitle="minutes of audio approved per day (last 30 days)" empty={!days.length && 'Nothing reviewed yet'}
          table={<DataTable columns={[{ key: 'day', label: 'day' }, { key: 'approved', label: 'approved', num: true }, { key: 'rejected', label: 'rejected', num: true }, { key: 'approved_hours', label: 'approved', num: true, fmt: hrs }]} rows={s.review_per_day} />}>
          <Columns data={days} color="var(--viz-series-1)" fmt={(v) => `${v.toFixed(v < 10 ? 1 : 0)}`} xLabelEvery={Math.ceil(days.length / 10)} />
        </ChartCard>

        <ChartCard title="Why pending clips are flagged" subtitle="a clip can carry several flags" empty={!flags.length && 'No flagged pending clips'}
          table={<DataTable columns={[{ key: 'title', label: 'flag' }, { key: 'value', label: 'clips', num: true }]} rows={flags} />}>
          <HBars data={flags} color="var(--viz-series-1)" fmt={(v) => v.toLocaleString()} />
        </ChartCard>
        <ChartCard title="Rejection reasons" subtitle="automatic (too_short, too_long, low_quality) and manual" empty={!rejects.length && 'No rejected clips'}
          table={<DataTable columns={[{ key: 'label', label: 'reason' }, { key: 'value', label: 'clips', num: true }]} rows={rejects} />}>
          <HBars data={rejects} color="var(--viz-series-1)" fmt={(v) => v.toLocaleString()} labelWidth={110} />
        </ChartCard>

        <EnginesCard health={health} />
        <section className="viz-root rounded-lg bg-[var(--viz-surface)] p-4 shadow-sm ring-1 ring-[var(--viz-border)]">
          <h2 className="text-sm font-semibold text-[var(--viz-text-1)]">Jobs in progress</h2>
          <p className="mt-0.5 mb-3 text-xs text-[var(--viz-text-3)]">processing, re-transcription and exports</p>
          {!jobs?.length ? (
            <div className="py-6 text-center text-sm text-[var(--viz-text-3)]">No jobs running</div>
          ) : (
            <ul className="space-y-3">
              {jobs.map((j) => (
                <li key={j.id}>
                  <div className="flex items-center gap-2 text-xs">
                    <StatusBadge status={j.status} />
                    <span className="font-medium text-[var(--viz-text-1)]">{j.type}</span>
                    <span className="text-[var(--viz-text-3)]">{j.source_file_id ? `file #${j.source_file_id}` : `export #${j.export_id}`}</span>
                    <span className="ml-auto truncate text-[var(--viz-text-2)]">{j.message}</span>
                  </div>
                  <ProgressBar className="mt-1.5" value={j.progress} />
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  )
}
