// Small SVG chart kit for the dashboard (dataviz method: thin marks, 4px rounded data-ends,
// 2px surface gaps, hairline axes, per-mark hover/focus tooltips, legend + table view).
import { useLayoutEffect, useRef, useState } from 'react'

export function useWidth() {
  const ref = useRef(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width]
}

// rectangle whose data-end corners are rounded (r) and whose baseline side is square
function barPath(x, y, w, h, r, horizontal) {
  r = Math.max(0, Math.min(r, horizontal ? w : h, horizontal ? h / 2 : w / 2))
  if (horizontal) {
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`
  }
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`
}

function niceMax(v) {
  if (v <= 0) return 1
  const p = 10 ** Math.floor(Math.log10(v))
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * p) return m * p
  return 10 * p
}

// ---------------------------------------------------------------- tooltip
function useTooltip() {
  const [tip, setTip] = useState(null)
  const show = (e, content) => {
    const box = e.currentTarget.ownerSVGElement?.parentElement?.getBoundingClientRect()
    const r = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX ?? r.left + r.width / 2) - (box?.left ?? 0)
    const y = (e.clientY ?? r.top) - (box?.top ?? 0)
    setTip({ x, y, content })
  }
  const node = tip && (
    <div
      className="pointer-events-none absolute z-20 min-w-36 -translate-x-1/2 -translate-y-[calc(100%+10px)] rounded-md bg-[var(--viz-tooltip)] px-2.5 py-1.5 text-xs shadow-lg ring-1 ring-black/5"
      style={{ left: tip.x, top: tip.y }}
    >
      {tip.content}
    </div>
  )
  return { show, hide: () => setTip(null), node }
}

function TipRow({ color, label, value, line = false }) {
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className={line ? 'h-0.5 w-3 rounded' : 'h-2.5 w-2.5 rounded-sm'} style={{ background: color }} />
      <span className="text-[var(--viz-text-2)]">{label}</span>
      <span className="ml-auto pl-3 font-semibold text-[var(--viz-text-1)] tabular-nums">{value}</span>
    </div>
  )
}

export function Legend({ keys }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--viz-text-2)]">
      {keys.map((k) => (
        <span key={k.key} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: k.color }} />
          {k.icon && <span aria-hidden className="text-[var(--viz-text-3)]">{k.icon}</span>}
          {k.label}
        </span>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- horizontal stacked bars
export function StackedBars({ rows, keys, fmt, labelWidth = 110 }) {
  const [ref, width] = useWidth()
  const tip = useTooltip()
  const bar = 18
  const gap = 14
  const totals = rows.map((r) => keys.reduce((s, k) => s + (r.values[k.key] || 0), 0))
  const max = Math.max(...totals, 0)
  const endLabel = 64
  const plotW = Math.max(40, width - labelWidth - endLabel)
  const scale = (v) => (max > 0 ? (v / max) * plotW : 0)
  const height = rows.length * (bar + gap) + 4
  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="stacked horizontal bars">
          {rows.map((r, i) => {
            const y = i * (bar + gap) + 2
            let x = labelWidth
            const present = keys.filter((k) => (r.values[k.key] || 0) > 0)
            return (
              <g key={r.label}>
                <text x={labelWidth - 10} y={y + bar / 2} dy="0.35em" textAnchor="end" className="fill-[var(--viz-text-2)] text-[12px]">
                  {r.label.length > 16 ? `${r.label.slice(0, 15)}…` : r.label}
                </text>
                <line x1={labelWidth} x2={labelWidth} y1={y - 3} y2={y + bar + 3} stroke="var(--viz-axis)" />
                {present.map((k, j) => {
                  const w = scale(r.values[k.key])
                  const last = j === present.length - 1
                  const segW = Math.max(0, w - (last ? 0 : 2)) // 2px surface gap between segments
                  const node = (
                    <path
                      key={k.key}
                      d={barPath(x, y, segW, bar, last ? 4 : 0, true)}
                      fill={k.color}
                      tabIndex={0}
                      aria-label={`${r.label} ${k.label}: ${fmt(r.values[k.key])}`}
                      className="cursor-default outline-none hover:opacity-80 focus:opacity-80"
                      onPointerMove={(e) => tip.show(e, <SegmentTip row={r} keys={keys} fmt={fmt} />)}
                      onFocus={(e) => tip.show(e, <SegmentTip row={r} keys={keys} fmt={fmt} />)}
                      onPointerLeave={tip.hide}
                      onBlur={tip.hide}
                    />
                  )
                  x += w
                  return node
                })}
                <text x={labelWidth + scale(totals[i]) + 6} y={y + bar / 2} dy="0.35em" className="fill-[var(--viz-text-1)] text-[12px] tabular-nums">
                  {fmt(totals[i])}
                </text>
              </g>
            )
          })}
        </svg>
      )}
      {tip.node}
    </div>
  )
}

function SegmentTip({ row, keys, fmt }) {
  return (
    <div>
      <div className="mb-1 font-medium text-[var(--viz-text-1)]">{row.label}</div>
      {keys.map((k) => (
        <TipRow key={k.key} color={k.color} label={k.label} value={fmt(row.values[k.key] || 0)} />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- columns (histogram / per day)
export function Columns({ data, color, fmt, height = 180, markers = [], tipLabel, xLabelEvery = 1 }) {
  const [ref, width] = useWidth()
  const tip = useTooltip()
  const left = 36
  const bottom = 22
  const plotH = height - bottom - 8
  const max = niceMax(Math.max(...data.map((d) => d.value), 0))
  const n = Math.max(data.length, 1)
  const slot = (width - left) / n
  const barW = Math.min(24, Math.max(2, slot - 2))
  const y = (v) => 8 + plotH - (v / max) * plotH
  const ticks = [0, max / 2, max]
  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="column chart">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={left} x2={width} y1={y(t)} y2={y(t)} stroke={t === 0 ? 'var(--viz-axis)' : 'var(--viz-grid)'} />
              <text x={left - 6} y={y(t)} dy="0.35em" textAnchor="end" className="fill-[var(--viz-text-3)] text-[10px] tabular-nums">
                {fmt(t)}
              </text>
            </g>
          ))}
          {data.map((d, i) => {
            const x = left + i * slot + (slot - barW) / 2
            const h = Math.max(0, y(0) - y(d.value))
            return (
              <g key={d.key ?? d.label}>
                {/* generous transparent hit target, bigger than the mark */}
                <rect
                  x={left + i * slot}
                  y={8}
                  width={slot}
                  height={plotH}
                  fill="transparent"
                  tabIndex={0}
                  aria-label={`${d.label}: ${fmt(d.value)}`}
                  className="outline-none"
                  onPointerMove={(e) => tip.show(e, d.tip ?? <TipRow color={color} label={`${tipLabel ?? ''}${d.label}`} value={fmt(d.value)} />)}
                  onFocus={(e) => tip.show(e, d.tip ?? <TipRow color={color} label={`${tipLabel ?? ''}${d.label}`} value={fmt(d.value)} />)}
                  onPointerLeave={tip.hide}
                  onBlur={tip.hide}
                />
                {h > 0 && <path d={barPath(x, y(d.value), barW, h, 4, false)} fill={color} pointerEvents="none" />}
                {d.edge !== undefined ? (
                  d.edge && (
                    <text x={left + i * slot} y={height - 6} textAnchor="middle" className="fill-[var(--viz-text-3)] text-[10px]">
                      {d.edge}
                    </text>
                  )
                ) : (
                  i % xLabelEvery === 0 && (
                    <text x={left + i * slot + slot / 2} y={height - 6} textAnchor="middle" className="fill-[var(--viz-text-3)] text-[10px]">
                      {d.short ?? d.label}
                    </text>
                  )
                )}
              </g>
            )
          })}
          {markers.map((m) => {
            const x = left + m.at * slot
            return (
              <g key={m.label}>
                <line x1={x} x2={x} y1={4} y2={y(0)} stroke="var(--viz-text-3)" />
                <text x={m.side === 'left' ? x - 4 : x + 4} y={14} textAnchor={m.side === 'left' ? 'end' : 'start'}
                  className="fill-[var(--viz-text-2)] text-[10px]">
                  {m.label}
                </text>
              </g>
            )
          })}
        </svg>
      )}
      {tip.node}
    </div>
  )
}

// ---------------------------------------------------------------- horizontal single-series bars
export function HBars({ data, color, fmt = (v) => v, labelWidth = 140 }) {
  const [ref, width] = useWidth()
  const tip = useTooltip()
  const bar = 16
  const gap = 10
  const max = Math.max(...data.map((d) => d.value), 0)
  const plotW = Math.max(40, width - labelWidth - 48)
  const height = data.length * (bar + gap) + 4
  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="bar chart">
          {data.map((d, i) => {
            const y = i * (bar + gap) + 2
            const w = max > 0 ? (d.value / max) * plotW : 0
            return (
              <g key={d.label}>
                <text x={labelWidth - 10} y={y + bar / 2} dy="0.35em" textAnchor="end" className="fill-[var(--viz-text-2)] text-[12px]">
                  {d.label}
                </text>
                <line x1={labelWidth} x2={labelWidth} y1={y - 2} y2={y + bar + 2} stroke="var(--viz-axis)" />
                <rect
                  x={labelWidth}
                  y={y - gap / 2}
                  width={plotW + 48}
                  height={bar + gap}
                  fill="transparent"
                  tabIndex={0}
                  aria-label={`${d.label}: ${fmt(d.value)}`}
                  className="outline-none"
                  onPointerMove={(e) => tip.show(e, <TipRow color={color} label={d.title ?? d.label} value={fmt(d.value)} />)}
                  onFocus={(e) => tip.show(e, <TipRow color={color} label={d.title ?? d.label} value={fmt(d.value)} />)}
                  onPointerLeave={tip.hide}
                  onBlur={tip.hide}
                />
                <path d={barPath(labelWidth, y, w, bar, 4, true)} fill={color} pointerEvents="none" />
                <text x={labelWidth + w + 6} y={y + bar / 2} dy="0.35em" className="fill-[var(--viz-text-1)] text-[12px] tabular-nums" pointerEvents="none">
                  {fmt(d.value)}
                </text>
              </g>
            )
          })}
        </svg>
      )}
      {tip.node}
    </div>
  )
}

// ---------------------------------------------------------------- card with a table-view twin
export function ChartCard({ title, subtitle, legend, table, children, empty }) {
  const [view, setView] = useState('chart')
  return (
    <section className="viz-root rounded-lg bg-[var(--viz-surface)] p-4 shadow-sm ring-1 ring-[var(--viz-border)]">
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[var(--viz-text-1)]">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-[var(--viz-text-3)]">{subtitle}</p>}
        </div>
        {table && !empty && (
          <div className="flex rounded-md bg-[var(--viz-page)] p-0.5 text-[11px] ring-1 ring-[var(--viz-border)]">
            {['chart', 'table'].map((v) => (
              <button key={v} onClick={() => setView(v)} className={`rounded px-2 py-0.5 ${view === v ? 'bg-[var(--viz-surface)] font-medium text-[var(--viz-text-1)] shadow-sm' : 'text-[var(--viz-text-3)]'}`}>
                {v}
              </button>
            ))}
          </div>
        )}
      </header>
      {empty ? (
        <div className="py-8 text-center text-sm text-[var(--viz-text-3)]">{empty}</div>
      ) : (
        <>
          {legend && view === 'chart' && <div className="mb-3">{legend}</div>}
          {view === 'chart' ? children : table}
        </>
      )}
    </section>
  )
}

export function DataTable({ columns, rows }) {
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-[var(--viz-surface)] text-left text-[var(--viz-text-3)]">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={`py-1 pr-3 font-medium ${c.num ? 'text-right' : ''}`}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--viz-grid)] text-[var(--viz-text-1)]">
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key} className={`py-1 pr-3 ${c.num ? 'text-right tabular-nums' : ''}`}>{c.fmt ? c.fmt(r[c.key]) : r[c.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
