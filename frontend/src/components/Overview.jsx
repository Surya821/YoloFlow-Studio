import { useEffect, useState } from 'react'
import { ArrowRight, Award, Boxes, Image as ImageIcon, PenSquare, Tag, Video } from 'lucide-react'
import { api, timeAgo } from '../lib/api'
import { Button, Card, EmptyState, Stat } from '../lib/ui'
import { MetricTiles } from './TrainingMonitor'

export default function Overview({ project, onNavigate, onOpenRun }) {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    api.stats(project.name).then(setStats).catch(() => {})
  }, [project.name])

  if (!stats) return <div className="p-8 text-sm text-slate-500">Loading…</div>
  const ds = stats.dataset
  const maxBoxes = Math.max(1, ...ds.per_class.map((c) => c.boxes))
  const run = stats.latest_run

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <header className="flex items-end justify-between">
          <div>
            <h1 className="text-xl font-semibold text-slate-50">{project.name}</h1>
            <p className="text-sm text-slate-500">Created {timeAgo(project.created_at)} · {stats.runs_total} training runs</p>
          </div>
          <div className="flex gap-2">
            <Button icon={Video} onClick={() => onNavigate('media')}>Add media</Button>
            <Button variant="primary" icon={PenSquare} onClick={() => onNavigate('annotate')}>Annotate</Button>
          </div>
        </header>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Stat label="Raw media" value={stats.media_count} icon={Video} sub="photos, videos, recordings" />
          <Stat label="Extracted frames" value={stats.frame_count} icon={ImageIcon} />
          <Stat label="Labeled images" value={ds.labeled_total} icon={PenSquare}
                sub={`${ds.images.train} train · ${ds.images.val} val · ${ds.background_images} background`} />
          <Stat label="Bounding boxes" value={ds.boxes_total} icon={Boxes} sub={`${ds.per_class.length} classes`} />
        </div>

        <div className="grid gap-6 lg:grid-cols-5">
          <Card title="Class distribution" icon={Tag} className="lg:col-span-2">
            {ds.per_class.length === 0 ? (
              <p className="text-sm text-slate-500">No classes yet — add them in the annotator.</p>
            ) : (
              <ul className="space-y-2.5">
                {ds.per_class.map((c) => (
                  <li key={c.id} title={`${c.name}: ${c.boxes} boxes`}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="flex items-center gap-1.5 text-slate-300">
                        <span className="size-2.5 rounded-sm" style={{ background: c.color }} />
                        {c.name}
                      </span>
                      <span className="tabular-nums text-slate-400">{c.boxes}</span>
                    </div>
                    <div className="h-2 rounded bg-slate-800">
                      <div className="h-full rounded" style={{ width: `${(c.boxes / maxBoxes) * 100}%`, background: c.color }} />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card
            title="Latest trained model"
            icon={Award}
            className="lg:col-span-3"
            actions={run && <Button size="sm" variant="ghost" onClick={() => onOpenRun(run.id)}>Open run <ArrowRight className="size-3.5" /></Button>}
          >
            {!run ? (
              <EmptyState icon={Award} title="No completed runs yet"
                          action={<Button variant="primary" onClick={() => onNavigate('train')}>Configure training</Button>}>
                Train on Kaggle GPUs or locally; metrics and plots will appear here.
              </EmptyState>
            ) : (
              <div className="space-y-4">
                <div className="text-xs text-slate-500">
                  {run.model} · {run.config.epochs} epochs · {run.backend} · {timeAgo(run.updated_at)}
                </div>
                <MetricTiles metrics={run.metrics} />
                {run.artifacts?.includes('results.png') && (
                  <img src={api.artifactUrl(project.name, run.id, 'results.png')} alt="Training results"
                       className="w-full rounded-lg border border-slate-800 bg-white" />
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
