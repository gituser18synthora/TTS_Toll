// Word-level diff (LCS) between two transcripts: removed words struck through red, added words green.

function tokens(text) {
  return (text || '').split(/\s+/).filter(Boolean)
}

const norm = (w) => w.toLowerCase().replace(/[\p{P}\p{S}]/gu, '')

export function diffWords(a, b) {
  const x = tokens(a)
  const y = tokens(b)
  const n = x.length
  const m = y.length
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1))
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = norm(x[i]) === norm(y[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  const ops = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (norm(x[i]) === norm(y[j])) {
      ops.push({ op: 'eq', text: y[j] })
      i++
      j++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push({ op: 'del', text: x[i++] })
    else ops.push({ op: 'ins', text: y[j++] })
  }
  while (i < n) ops.push({ op: 'del', text: x[i++] })
  while (j < m) ops.push({ op: 'ins', text: y[j++] })
  return ops
}

export default function WordDiff({ from, to, className = '' }) {
  const ops = diffWords(from, to)
  if (!ops.length) return <span className="text-slate-400 italic">empty</span>
  return (
    <span className={className} dir="auto">
      {ops.map((o, k) => (
        <span key={k}>
          {o.op === 'eq' && o.text}
          {o.op === 'del' && <del className="rounded bg-rose-100 px-0.5 text-rose-700 decoration-rose-400">{o.text}</del>}
          {o.op === 'ins' && <ins className="rounded bg-emerald-100 px-0.5 text-emerald-800 no-underline">{o.text}</ins>}{' '}
        </span>
      ))}
    </span>
  )
}
