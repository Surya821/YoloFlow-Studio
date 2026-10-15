import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Activity, AlertCircle, CheckCircle2, Circle, Cloud, Download, ExternalLink, FileText, Image as ImageIcon, Laptop,
  LineChart as LineIcon, Loader2, RefreshCw, Square, Trash2,
} from 'lucide-react'
import { api, formatDuration, timeAgo } from '../lib/api'
import { Badge, Button, Card, EmptyState, Progress, Segmented, cx, confirmAction } from '../lib/ui'

const ACTIVE = new Set(['preparing', 'queued', 'running', 'downloading'])

const STATUS = {
  preparing: { tone: 'amber', label: 'Preparing', pulse: true },
  queued: { tone: 'amber', label: 'Queued', pulse: true },
  running: { tone: 'amber', label: 'Running', pulse: true },
  downloading: { tone: 'amber', label: 'Downloading', pulse: true },
  completed: { tone: 'emerald', label: 'Completed' },
  error: { tone: 'rose', label: 'Error' },
  cancelled: { tone: 'slate', label: 'Cancelled' },
}

const STAGES = {
  kaggle: [
    ['packaging', 'Package dataset'],
    ['uploading_dataset', 'Upload to Kaggle'],
    ['processing_dataset', 'Kaggle processing'],
    ['pushing_kernel', 'Push kernel'],
    ['queued', 'Queued'],
    ['running', 'Training on GPU'],
    ['downloading', 'Download artifacts'],
    ['completed', 'Completed'],
  ],
  local: [
    ['running', 'Training locally'],
    ['completed', 'Completed'],
  ],
}

// Validated (dark surface #0F172A): blue / orange pass lightness, chroma, CVD and contrast checks.
const SERIES = { train: '#3987e5', val: '#d95926' }

export function StatusBadge({ status }) {
  const s = STATUS[status] || STATUS.preparing
  return <Badge tone={s.tone} dot pulse={s.pulse}>{s.label}</Badge>
}

export function MetricTiles({ metrics }) {
  const items = [
    ['mAP50', metrics?.map50],
    ['mAP50-95', metrics?.map50_95],
    ['Precision', metrics?.precision],
    ['Recall', metrics?.recall],
  ]
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map(([label, v]) => (
        <div key={label} className="rounded-lg border border-slate-800 bg-slate-950 p-3">
          <div className="text-[11px] font-medium text-slate-400">{label}</div>
          <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-50">{v == null ? '—' : v.toFixed(3)}</div>
          <div className="mt-2 h-1 rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-emerald-500" style={{ width: `${(v || 0) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function TrainingMonitor({ project, focusRun, onFocusRun }) {
  const [runs, setRuns] = useState([])
  const [loaded, setLoaded] = useState(false)
  const selectedId = focusRun && runs.some((r) => r.id === focusRun) ? focusRun : runs[0]?.id
  const anyActive = runs.some((r) => ACTIVE.has(r.status))

  const loadRuns = async () => {
    try {
      setRuns(await api.runs(project.name))
    } catch (e) {
      toast.error(e.message)
    } finally {
      setLoaded(true)
    }
  }

  useEffect(() => {
    loadRuns()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.name])

  useEffect(() => {
    if (!anyActive) return
    const t = setInterval(loadRuns, 5000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anyActive, project.name])

  if (loaded && !runs.length) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <EmptyState icon={Activity} title="No training runs yet">Configure and launch a run from the Train tab.</EmptyState>
      </div>
    )
  }

  const run = runs.find((r) => r.id === selectedId)

  return (
    <div className="flex h-full">
      <aside className="w-72 shrink-0 overflow-y-auto border-r border-slate-800 bg-slate-950 p-2">
        <div className="px-2 pb-2 pt-1 text-xs font-medium uppercase tracking-wider text-slate-500">Runs</div>
        {runs.map((r) => (
          <button key={r.id} onClick={() => onFocusRun(r.id)}
                  className={cx('mb-1 w-full rounded-lg p-2.5 text-left transition-colors',
                    r.id === selectedId ? 'bg-slate-900 ring-1 ring-slate-800' : 'hover:bg-slate-900/60')}>
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 truncate font-mono text-xs text-slate-200">
                {r.backend === 'kaggle' ? <Cloud className="size-3.5 text-sky-400" /> : <Laptop className="size-3.5 text-slate-400" />}
                {r.model}
              </span>
              <StatusBadge status={r.status} />
            </div>
            <div className="mt-1 flex justify-between text-[11px] text-slate-500">
              <span>{r.config.epochs} ep · {r.config.imgsz}px</span>
              <span>{timeAgo(r.created_at)}</span>
            </div>
            {r.metrics?.map50 != null && (
              <div className="mt-1 text-[11px] text-slate-400">mAP50 <span className="tabular-nums text-emerald-400">{r.metrics.map50.toFixed(3)}</span></div>
            )}
          </button>
        ))}
      </aside>
      <div className="min-w-0 flex-1 overflow-y-auto">
        {run && <RunDetail key={run.id} project={project} run={run} onChanged={loadRuns} />}
      </div>
    </div>
  )
}

function RunDetail({ project, run, onChanged }) {
  const [log, setLog] = useState('')
  const [results, setResults] = useState({ columns: [], rows: [] })
  const [view, setView] = useState('charts')
  const [autoScroll, setAutoScroll] = useState(true)
  const logRef = useRef(null)
  const active = ACTIVE.has(run.status)

  const loadDetail = async () => {
    try {
      const [l, r] = await Promise.all([api.runLog(project.name, run.id), api.runResults(project.name, run.id)])
      setLog(l.log)
      setResults(r)
    } catch {
      /* transient */
    }
  }

  useEffect(() => {
    loadDetail()
    if (!active) return
    const t = setInterval(loadDetail, 4000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.id, active, run.status])

  useEffect(() => {
    if (autoScroll && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log, autoScroll])

  const cancel = async () => {
    const msg = run.backend === 'kaggle'
      ? 'Stop tracking this Kaggle run? (The Kaggle API cannot cancel kernels — stop it on kaggle.com if needed.)'
      : 'Stop local training?'
    if (!confirmAction(msg)) return
    await api.cancelRun(project.name, run.id)
    onChanged()
  }

  const remove = async () => {
    if (!confirmAction('Delete this run and its downloaded weights?')) return
    await api.deleteRun(project.name, run.id)
    onChanged()
    toast.success('Run deleted')
  }

  const stages = STAGES[run.backend]
  const reached = new Set((run.timeline || []).map((t) => t.stage))
  const currentIdx = stages.findIndex(([s]) => s === run.stage)
  const epoch = run.progress?.epoch || 0
  const epochs = run.progress?.epochs || run.config.epochs
  const elapsed = (run.status === 'completed' || run.status === 'error' ? run.updated_at : Date.now() / 1000) - run.created_at
  const plots = (run.artifacts || []).filter((a) => /\.(png|jpg)$/i.test(a))
  const downloads = (run.artifacts || []).filter((a) => /\.(pt|onnx|csv|json|yaml)$/i.test(a))

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-mono text-lg font-semibold text-slate-50">{run.model}</h1>
            <StatusBadge status={run.status} />
          </div>
          <p className="mt-1 text-sm text-slate-500">
            {run.id} · {run.backend === 'kaggle' ? 'Kaggle GPU' : 'Local'} · {run.config.epochs} epochs · batch {run.config.batch} ·
            {' '}{run.config.imgsz}px · {run.config.optimizer} · lr0 {run.config.lr0} · {formatDuration(elapsed)}
          </p>
        </div>
        <div className="flex gap-2">
          {run.kaggle?.kernel_url && (
            <a href={run.kaggle.kernel_url} target="_blank" rel="noreferrer">
              <Button size="sm" icon={ExternalLink}>Kaggle kernel</Button>
            </a>
          )}
          {run.backend === 'kaggle' && active && (
            <Button size="sm" variant="ghost" icon={RefreshCw} onClick={() => api.refreshRun(project.name, run.id).then(onChanged)}>Re-poll</Button>
          )}
          {active && <Button size="sm" variant="dangerGhost" icon={Square} onClick={cancel}>Stop</Button>}
          {!active && <Button size="sm" variant="dangerGhost" icon={Trash2} onClick={remove}>Delete</Button>}
        </div>
      </header>

      {run.error && (
        <div className="flex gap-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
          <AlertCircle className="mt-0.5 size-4 shrink-0" /> {run.error}
        </div>
      )}
      {run.poll_error && active && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300">
          Last poll failed, retrying: {run.poll_error}
        </div>
      )}

      <Card title="Pipeline" icon={Activity} bodyClass="space-y-4">
        <ol className="flex flex-wrap gap-x-1 gap-y-2">
          {stages.map(([id, label], i) => {
            const done = run.status === 'completed' || (reached.has(id) && i < currentIdx)
            const isCurrent = i === currentIdx && run.status !== 'completed'
            const failed = isCurrent && run.status === 'error'
            return (
              <li key={id} className="flex items-center gap-1">
                <span className={cx('flex items-center gap-1.5 rounded-md px-2 py-1 text-xs',
                  failed ? 'bg-rose-500/10 text-rose-300' : isCurrent ? 'bg-amber-500/10 text-amber-300' : done ? 'text-emerald-400' : 'text-slate-500')}>
                  {failed ? <AlertCircle className="size-3.5" /> : isCurrent ? <Loader2 className="size-3.5 animate-spin" />
                    : done ? <CheckCircle2 className="size-3.5" /> : <Circle className="size-3.5" />}
                  {label}
                </span>
                {i < stages.length - 1 && <span className="text-slate-700">›</span>}
              </li>
            )
          })}
        </ol>
        <div>
          <div className="mb-1.5 flex justify-between text-xs text-slate-400">
            <span>Epoch progress</span>
            <span className="tabular-nums">{epoch} / {epochs}</span>
          </div>
          <Progress value={run.status === 'completed' ? 1 : epoch / epochs} tone={run.status === 'error' ? 'rose' : active ? 'amber' : 'emerald'} />
          {run.backend === 'kaggle' && active && (
            <p className="mt-2 text-[11px] text-slate-500">
              Kaggle status polled every 15 s{run.kaggle_status ? ` · kernel ${run.kaggle_status.toLowerCase()}` : ''}
              {run.last_polled ? ` · last check ${timeAgo(run.last_polled)}` : ''}. Live logs depend on Kaggle — full log arrives on completion.
            </p>
          )}
        </div>
      </Card>

      {(run.metrics || results.rows.length > 0) && (
        <Card title="Results" icon={LineIcon}
              actions={<Segmented size="sm" value={view} onChange={setView}
                                  options={[{ value: 'charts', label: 'Charts' }, { value: 'table', label: 'Table' }]} />}
              bodyClass="space-y-5">
          <MetricTiles metrics={run.metrics || lastMetrics(results)} />
          {view === 'charts' ? <TrainingCharts results={results} /> : <ResultsTable results={results} />}
        </Card>
      )}

      {plots.length > 0 && (
        <Card title="Plots" icon={ImageIcon}>
          <div className="grid gap-4 md:grid-cols-2">
            {plots.map((p) => (
              <a key={p} href={api.artifactUrl(project.name, run.id, p)} target="_blank" rel="noreferrer" className="group block">
                <img src={api.artifactUrl(project.name, run.id, p)} alt={p} loading="lazy"
                     className="w-full rounded-lg border border-slate-800 bg-white transition group-hover:border-slate-600" />
                <span className="mt-1 block text-[11px] text-slate-500">{p}</span>
              </a>
            ))}
          </div>
        </Card>
      )}

      {downloads.length > 0 && (
        <Card title="Artifacts" icon={Download}>
          <div className="flex flex-wrap gap-2">
            {downloads.map((a) => (
              <a key={a} href={api.artifactUrl(project.name, run.id, a)} download>
                <Button size="sm" variant={/best\.(pt|onnx)$/.test(a) ? 'primary' : 'secondary'} icon={Download}>{a}</Button>
              </a>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-slate-500">
            Stored in workspace/projects/{project.name}/outputs/{run.id}/artifacts/
          </p>
        </Card>
      )}

      <Card title="Log" icon={FileText}
            actions={
              <label className="flex items-center gap-1.5 text-xs text-slate-400">
                <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} className="accent-emerald-600" />
                Auto-scroll
              </label>
            }
            bodyClass="p-0">
        <pre ref={logRef} className="h-96 overflow-auto whitespace-pre-wrap break-all bg-slate-950 p-4 font-mono text-[11px] leading-relaxed text-slate-300">
          {log || (active ? 'Waiting for output…' : 'No log output.')}
        </pre>
      </Card>
    </div>
  )
}

function lastMetrics(results) {
  if (!results.rows.length) return null
  const last = results.rows[results.rows.length - 1]
  const get = (name) => {
    const i = results.columns.indexOf(name)
    return i >= 0 ? last[i] : null
  }
  return {
    map50: get('metrics/mAP50(B)'),
    map50_95: get('metrics/mAP50-95(B)'),
    precision: get('metrics/precision(B)'),
    recall: get('metrics/recall(B)'),
  }
}

// ------------------------------------------------------------------ charts

function TrainingCharts({ results }) {
  const { columns, rows } = results
  const charts = useMemo(() => {
    if (!rows.length) return []
    const col = (name) => {
      const i = columns.indexOf(name)
      return i >= 0 ? rows.map((r) => r[i]) : null
    }
    const epochIdx = columns.indexOf('epoch')
    const x = epochIdx >= 0 ? rows.map((r) => r[epochIdx]) : rows.map((_, i) => i + 1)
    // One chart per loss: train vs val share a unit, so one axis each.
    const losses = columns
      .filter((c) => c.startsWith('train/') && c.endsWith('_loss'))
      .map((c) => c.slice(6))
      .map((loss) => ({
        title: loss.replace('_loss', ' loss'),
        x,
        series: [
          { name: 'train', color: SERIES.train, values: col(`train/${loss}`) },
          { name: 'val', color: SERIES.val, dash: '5 4', values: col(`val/${loss}`) },
        ].filter((s) => s.values),
      }))
    const maps = {
      title: 'mAP (validation)',
      x,
      series: [
        { name: 'mAP50', color: SERIES.train, values: col('metrics/mAP50(B)') },
        { name: 'mAP50-95', color: SERIES.val, dash: '5 4', values: col('metrics/mAP50-95(B)') },
      ].filter((s) => s.values),
    }
    return [maps, ...losses].filter((c) => c.series.length)
  }, [columns, rows])

  if (!charts.length) return <p className="text-xs text-slate-500">Per-epoch results appear once the first epoch finishes.</p>
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {charts.map((c) => <LineChart key={c.title} {...c} />)}
    </div>
  )
}

function LineChart({ title, x, series, height = 180 }) {
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(400)
  const [hover, setHover] = useState(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const pad = { l: 44, r: 12, t: 10, b: 24 }
  const iw = width - pad.l - pad.r
  const ih = height - pad.t - pad.b
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v))
  let lo = Math.min(...all)
  let hi = Math.max(...all)
  if (lo === hi) {
    lo -= 0.5
    hi += 0.5
  }
  const span = hi - lo
  lo = Math.max(lo >= 0 ? 0 : -Infinity, lo - span * 0.08)
  hi += span * 0.08
  const xMin = x[0]
  const xMax = x[x.length - 1] === xMin ? xMin + 1 : x[x.length - 1]
  const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * iw
  const sy = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * ih
  const ticks = Array.from({ length: 4 }, (_, i) => lo + ((hi - lo) * i) / 3)
  const fmt = (v) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(Math.abs(v) >= 1 ? 2 : 3))

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    let best = 0
    for (let i = 1; i < x.length; i++) if (Math.abs(sx(x[i]) - px) < Math.abs(sx(x[best]) - px)) best = i
    setHover(best)
  }

  return (
    <figure className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <figcaption className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium capitalize text-slate-300">{title}</span>
        <span className="flex gap-3">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <svg width="16" height="6" aria-hidden><line x1="0" y1="3" x2="16" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} /></svg>
              {s.name}
            </span>
          ))}
        </span>
      </figcaption>
      <div ref={wrapRef} className="relative">
        <svg width={width} height={height} onMouseMove={onMove} onMouseLeave={() => setHover(null)} className="block">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} stroke="#1e293b" strokeWidth="1" />
              <text x={pad.l - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="#64748b">{fmt(t)}</text>
            </g>
          ))}
          <text x={pad.l} y={height - 6} fontSize="10" fill="#64748b">{xMin}</text>
          <text x={width - pad.r} y={height - 6} fontSize="10" fill="#64748b" textAnchor="end">epoch {xMax}</text>
          {series.map((s) => {
            const d = s.values.map((v, i) => (Number.isFinite(v) ? `${i === 0 ? 'M' : 'L'}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}` : '')).join('')
            return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round" />
          })}
          {hover != null && (
            <g>
              <line x1={sx(x[hover])} x2={sx(x[hover])} y1={pad.t} y2={pad.t + ih} stroke="#475569" strokeWidth="1" />
              {series.map((s) => Number.isFinite(s.values[hover]) && (
                <circle key={s.name} cx={sx(x[hover])} cy={sy(s.values[hover])} r="4" fill={s.color} stroke="#020617" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {hover != null && (
          <div className="pointer-events-none absolute top-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-[11px] shadow-lg"
               style={{ left: Math.min(sx(x[hover]) + 10, width - 130) }}>
            <div className="mb-0.5 text-slate-400">Epoch {x[hover]}</div>
            {series.map((s) => (
              <div key={s.name} className="flex items-center gap-2 text-slate-200">
                <span className="size-2 rounded-full" style={{ background: s.color }} />
                {s.name} <span className="ml-auto tabular-nums">{Number.isFinite(s.values[hover]) ? fmt(s.values[hover]) : '—'}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}

function ResultsTable({ results }) {
  const keep = results.columns
    .map((c, i) => [c, i])
    .filter(([c]) => c === 'epoch' || c.includes('loss') || c.startsWith('metrics/'))
  if (!results.rows.length) return <p className="text-xs text-slate-500">No per-epoch results yet.</p>
  return (
    <div className="max-h-96 overflow-auto rounded-lg border border-slate-800">
      <table className="w-full text-left text-[11px]">
        <thead className="sticky top-0 bg-slate-900 text-slate-400">
          <tr>{keep.map(([c]) => <th key={c} className="whitespace-nowrap px-2.5 py-2 font-medium">{c.replace('metrics/', '')}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-800 font-mono tabular-nums text-slate-300">
          {results.rows.map((r, ri) => (
            <tr key={ri} className="hover:bg-slate-900">
              {keep.map(([c, i]) => <td key={c} className="px-2.5 py-1.5">{c === 'epoch' ? r[i] : r[i]?.toFixed(4)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


