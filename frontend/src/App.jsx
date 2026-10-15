import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from 'react-router-dom'
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

function ProjectTabRedirect({ projects }) {
  const { projectName } = useParams()
  const exists = projects.some((p) => p.name === projectName)
  if (!exists && projects.length) {
    return <Navigate to={`/projects/${encodeURIComponent(projects[0].name)}/overview`} replace />
  }
  return <Navigate to={`/projects/${encodeURIComponent(projectName)}/overview`} replace />
}

function ShortcutRedirect({ tab, projects, loading }) {
  if (loading) return null
  if (!projects.length) return <Navigate to="/" replace />
  const lastProject = store.get('yf.project', '')
  const target = projects.find((p) => p.name === lastProject)?.name || projects[0].name
  return <Navigate to={`/projects/${encodeURIComponent(target)}/${tab}`} replace />
}

function RootRedirect({ projects, loading, setCreating }) {
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="text-sm text-slate-500">Loading projects...</div>
      </div>
    )
  }
  if (!projects.length) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <EmptyState
          icon={FolderPlus}
          title="Create your first project"
          action={
            <Button variant="primary" icon={Plus} onClick={() => setCreating(true)}>
              New project
            </Button>
          }
        >
          A project holds media, extracted frames, annotations, the YOLO dataset and trained weights.
        </EmptyState>
      </div>
    )
  }
  const lastProject = store.get('yf.project', '')
  const target = projects.find((p) => p.name === lastProject)?.name || projects[0].name
  const lastTab = store.get('yf.tab', 'overview')
  return <Navigate to={`/projects/${encodeURIComponent(target)}/${lastTab}`} replace />
}

function ProjectTabView({
  projects,
  catalog,
  tab,
  loadProjects,
}) {
  const { projectName, runId } = useParams()
  const navigate = useNavigate()
  const project = projects.find((p) => p.name === projectName)

  useEffect(() => {
    if (projectName) {
      store.set('yf.project', projectName)
    }
    if (tab) {
      store.set('yf.tab', tab)
    }
  }, [projectName, tab])

  if (!project) {
    if (projects.length) {
      return <Navigate to={`/projects/${encodeURIComponent(projects[0].name)}/overview`} replace />
    }
    return <Navigate to="/" replace />
  }

  const navigateTab = (newTab) => {
    navigate(`/projects/${encodeURIComponent(project.name)}/${newTab}`)
  }

  const openRun = (targetRunId) => {
    navigate(`/projects/${encodeURIComponent(project.name)}/monitor/${encodeURIComponent(targetRunId)}`)
  }

  return (
    <div className="h-full" key={project.name}>
      {tab === 'overview' && <Overview project={project} onNavigate={navigateTab} onOpenRun={openRun} />}
      {tab === 'media' && <MediaCapture project={project} onChanged={loadProjects} onAnnotate={() => navigateTab('annotate')} />}
      {tab === 'annotate' && <Annotator project={project} catalog={catalog} onChanged={loadProjects} />}
      {tab === 'train' && <TrainConfig project={project} catalog={catalog} onLaunched={openRun} />}
      {tab === 'monitor' && <TrainingMonitor project={project} focusRun={runId} onFocusRun={openRun} />}
      {tab === 'test' && <Inference project={project} catalog={catalog} />}
    </div>
  )
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [catalog, setCatalog] = useState(null)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
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
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadProjects()
    api.catalog().then(setCatalog).catch(() => {})
  }, [loadProjects])

  // Extract current project name & tab from URL
  const match = location.pathname.match(/^\/projects\/([^/]+)(?:\/([^/]+))?/)
  const routeProjectName = match ? decodeURIComponent(match[1]) : null
  const currentTab = match ? match[2] || 'overview' : store.get('yf.tab', 'overview')

  const currentProject = useMemo(() => {
    if (routeProjectName) {
      const found = projects.find((p) => p.name === routeProjectName)
      if (found) return found
    }
    const lastProject = store.get('yf.project', '')
    return projects.find((p) => p.name === lastProject) || projects[0] || null
  }, [routeProjectName, projects])

  const createProject = async () => {
    try {
      const p = await api.createProject(newName)
      await loadProjects()
      setCreating(false)
      setNewName('')
      toast.success(`Project "${p.name}" created`)
      navigate(`/projects/${encodeURIComponent(p.name)}/media`)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const deleteProject = async (name) => {
    if (!confirmAction(`Delete project "${name}" and ALL its media, labels and outputs? This cannot be undone.`)) return
    try {
      await api.deleteProject(name)
      const list = await loadProjects()
      toast.success('Project deleted')
      if (list.length) {
        navigate(`/projects/${encodeURIComponent(list[0].name)}/overview`)
      } else {
        navigate('/')
      }
    } catch (e) {
      toast.error(e.message)
    }
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
              <span className="block truncate text-sm font-medium text-slate-100">{currentProject?.name || 'No project'}</span>
            </span>
            <ChevronDown className="size-4 text-slate-500" />
          </button>
          {menuOpen && (
            <div className="absolute inset-x-3 top-full z-40 mt-1 rounded-lg border border-slate-800 bg-slate-900 p-1 shadow-xl">
              <div className="max-h-64 overflow-y-auto">
                {projects.map((p) => (
                  <div key={p.name} className="group flex items-center rounded-md hover:bg-slate-800">
                    <button
                      className={cx(
                        'flex-1 truncate px-2.5 py-1.5 text-left text-sm',
                        p.name === currentProject?.name ? 'text-emerald-400' : 'text-slate-300',
                      )}
                      onClick={() => {
                        setMenuOpen(false)
                        navigate(`/projects/${encodeURIComponent(p.name)}/${currentTab || 'overview'}`)
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
          {TABS.map((t) => {
            const isSelected = currentTab === t.id && Boolean(currentProject)
            return (
              <button
                key={t.id}
                disabled={!currentProject}
                onClick={() => {
                  if (currentProject) {
                    navigate(`/projects/${encodeURIComponent(currentProject.name)}/${t.id}`)
                  }
                }}
                className={cx(
                  'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40',
                  isSelected
                    ? 'bg-slate-900 text-white ring-1 ring-slate-800'
                    : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200',
                )}
              >
                <t.icon className={cx('size-4', isSelected && 'text-emerald-400')} />
                {t.label}
              </button>
            )
          })}
        </nav>

        {currentProject && (
          <div className="m-3 grid grid-cols-3 gap-1 rounded-lg border border-slate-800 bg-slate-900 p-2 text-center">
            {[
              ['Media', currentProject.media_count],
              ['Frames', currentProject.frame_count],
              ['Labeled', currentProject.labeled_count],
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
        ) : (
          <Routes>
            <Route path="/" element={<RootRedirect projects={projects} loading={loading} setCreating={setCreating} />} />
            <Route path="/overview" element={<ShortcutRedirect tab="overview" projects={projects} loading={loading} />} />
            <Route path="/media" element={<ShortcutRedirect tab="media" projects={projects} loading={loading} />} />
            <Route path="/annotate" element={<ShortcutRedirect tab="annotate" projects={projects} loading={loading} />} />
            <Route path="/train" element={<ShortcutRedirect tab="train" projects={projects} loading={loading} />} />
            <Route path="/monitor" element={<ShortcutRedirect tab="monitor" projects={projects} loading={loading} />} />
            <Route path="/test" element={<ShortcutRedirect tab="test" projects={projects} loading={loading} />} />

            <Route path="/projects/:projectName" element={<ProjectTabRedirect projects={projects} />} />
            <Route
              path="/projects/:projectName/overview"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="overview" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/media"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="media" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/annotate"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="annotate" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/train"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="train" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/monitor"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="monitor" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/monitor/:runId"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="monitor" loadProjects={loadProjects} />}
            />
            <Route
              path="/projects/:projectName/test"
              element={<ProjectTabView projects={projects} catalog={catalog} tab="test" loadProjects={loadProjects} />}
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
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
