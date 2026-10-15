import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  Activity, Boxes, ChevronDown, Cpu, FolderPlus, LayoutDashboard, PenSquare, Plus, ScanSearch, Trash2, Video,
} from 'lucide-react'
import { api } from './lib/api'
import { Button, EmptyState, Input, Modal, cx, confirmAction } from './lib/ui'
import Overview from './components/Overview'
import MediaCapture from './components/MediaCapture'
import Annotator from './components/Annotator'
import TrainConfig from './components/TrainConfig'
import TrainingMonitor from './components/TrainingMonitor'
import Inference from './components/Inference'

const TABS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'media', label: 'Media & Capture', icon: Video },
  { id: 'annotate', label: 'Annotate', icon: PenSquare },
  { id: 'train', label: 'Train', icon: Cpu },
  { id: 'monitor', label: 'Monitor', icon: Activity },
  { id: 'test', label: 'Test Model', icon: ScanSearch },
]

const store = {
  get: (k, d) => {
    try {
      return localStorage.getItem(k) ?? d
    } catch {
      return d
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v)
    } catch {
      /* ignore */
    }
  },
}

export default function App() {
  const [projects, setProjects] = useState([])
  const [current, setCurrent] = useState(() => store.get('yf.project', ''))
  const [tab, setTab] = useState(() => store.get('yf.tab', 'overview'))
  const [catalog, setCatalog] = useState(null)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [focusRun, setFocusRun] = useState(null)
  const [backendDown, setBackendDown] = useState(false)

  const loadProjects = useCallback(async () => {
    try {
      const list = await api.projects()
      setProjects(list)
      setBackendDown(false)
      return list
    } catch (e) {
      setBackendDown(true)
      return []
    }
  }, [])

  useEffect(() => {
    loadProjects().then((list) => {
      if (list.length && !list.find((p) => p.name === store.get('yf.project', ''))) setCurrent(list[0].name)
    })
    api.catalog().then(setCatalog).catch(() => {})
  }, [loadProjects])

  useEffect(() => store.set('yf.project', current || ''), [current])
  useEffect(() => store.set('yf.tab', tab), [tab])

  const project = projects.find((p) => p.name === current) || null

  const createProject = async () => {
    try {
      const p = await api.createProject(newName)
      await loadProjects()
      setCurrent(p.name)
      setTab('media')
      setCreating(false)
      setNewName('')
      toast.success(`Project "${p.name}" created`)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const deleteProject = async (name) => {
    if (!confirmAction(`Delete project "${name}" and ALL its media, labels and outputs? This cannot be undone.`)) return
    try {
      await api.deleteProject(name)
      const list = await loadProjects()
      setCurrent(list[0]?.name || '')
      toast.success('Project deleted')
    } catch (e) {
      toast.error(e.message)
    }
  }

  const goToRun = (runId) => {
    setFocusRun(runId)
    setTab('monitor')
  }

  return (
    <div className="flex h-full bg-slate-950 text-slate-200">
      {/* ------------------------------------------------ sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-950">
        <div className="flex items-center gap-2.5 px-4 py-4">
          <div className="flex size-8 items-center justify-center rounded-lg bg-emerald-600">
            <Boxes className="size-4.5 text-white" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight text-slate-50">YOLOFlow Studio</div>
            <div className="text-[11px] text-slate-500">Custom detector training</div>
          </div>
        </div>

        <div className="relative px-3 pb-3">
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-left hover:border-slate-700"
          >
            <span className="min-w-0">
              <span className="block text-[10px] uppercase tracking-wider text-slate-500">Project</span>
              <span className="block truncate text-sm font-medium text-slate-100">{project?.name || 'No project'}</span>
            </span>
            <ChevronDown className="size-4 text-slate-500" />
          </button>
          {menuOpen && (
            <div className="absolute inset-x-3 top-full z-40 mt-1 rounded-lg border border-slate-800 bg-slate-900 p-1 shadow-xl">
              <div className="max-h-64 overflow-y-auto">
                {projects.map((p) => (
                  <div key={p.name} className="group flex items-center rounded-md hover:bg-slate-800">
                    <button
                      className={cx('flex-1 truncate px-2.5 py-1.5 text-left text-sm', p.name === current ? 'text-emerald-400' : 'text-slate-300')}
                      onClick={() => {
                        setCurrent(p.name)
                        setMenuOpen(false)
                      }}
                    >
                      {p.name}
                    </button>
                    <button
                      title="Delete project"
                      onClick={() => {
                        setMenuOpen(false)
                        deleteProject(p.name)
                      }}
                      className="mr-1 rounded p-1 text-slate-500 opacity-0 hover:bg-rose-600/20 hover:text-rose-400 group-hover:opacity-100"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                ))}
              </div>
              <button
                onClick={() => {
                  setMenuOpen(false)
                  setCreating(true)
                }}
                className="mt-1 flex w-full items-center gap-2 rounded-md border-t border-slate-800 px-2.5 py-2 text-sm text-emerald-400 hover:bg-slate-800"
              >
                <Plus className="size-4" /> New project
              </button>
            </div>
          )}
        </div>

        <nav className="flex-1 space-y-0.5 px-3">
          {TABS.map((t) => (
            <button
              key={t.id}
              disabled={!project}
              onClick={() => setTab(t.id)}
              className={cx(
                'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40',
                tab === t.id && project ? 'bg-slate-900 text-white ring-1 ring-slate-800' : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200',
              )}
            >
              <t.icon className={cx('size-4', tab === t.id && project && 'text-emerald-400')} />
              {t.label}
            </button>
          ))}
        </nav>

        {project && (
          <div className="m-3 grid grid-cols-3 gap-1 rounded-lg border border-slate-800 bg-slate-900 p-2 text-center">
            {[
              ['Media', project.media_count],
              ['Frames', project.frame_count],
              ['Labeled', project.labeled_count],
            ].map(([l, v]) => (
              <div key={l}>
                <div className="text-sm font-semibold tabular-nums text-slate-100">{v}</div>
                <div className="text-[10px] text-slate-500">{l}</div>
              </div>
            ))}
          </div>
        )}
      </aside>

      {/* ------------------------------------------------ main */}
      <main className="min-w-0 flex-1 overflow-hidden">
        {backendDown ? (
          <div className="flex h-full items-center justify-center p-8">
            <EmptyState icon={Activity} title="Backend not reachable">
              Start the API with <code className="text-emerald-400">uvicorn main:app --port 8000</code> from the backend folder, then reload.
            </EmptyState>
          </div>
        ) : !project ? (
          <div className="flex h-full items-center justify-center p-8">
            <EmptyState
              icon={FolderPlus}
              title="Create your first project"
              action={<Button variant="primary" icon={Plus} onClick={() => setCreating(true)}>New project</Button>}
            >
              A project holds media, extracted frames, annotations, the YOLO dataset and trained weights.
            </EmptyState>
          </div>
        ) : (
          <div className="h-full" key={project.name}>
            {tab === 'overview' && <Overview project={project} onNavigate={setTab} onOpenRun={goToRun} />}
            {tab === 'media' && <MediaCapture project={project} onChanged={loadProjects} onAnnotate={() => setTab('annotate')} />}
            {tab === 'annotate' && <Annotator project={project} catalog={catalog} onChanged={loadProjects} />}
            {tab === 'train' && <TrainConfig project={project} catalog={catalog} onLaunched={goToRun} />}
            {tab === 'monitor' && <TrainingMonitor project={project} focusRun={focusRun} onFocusRun={setFocusRun} />}
            {tab === 'test' && <Inference project={project} catalog={catalog} />}
          </div>
        )}
      </main>

      <Modal
        open={creating}
        onClose={() => setCreating(false)}
        title="New project"
        footer={
          <>
            <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
            <Button variant="primary" onClick={createProject} disabled={!newName.trim()}>Create</Button>
          </>
        }
      >
        <Input
          autoFocus
          placeholder="e.g. construction-site-ppe"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && newName.trim() && createProject()}
        />
        <p className="mt-2 text-xs text-slate-500">Letters, digits, dashes and underscores. Spaces become dashes.</p>
      </Modal>
    </div>
  )
}
