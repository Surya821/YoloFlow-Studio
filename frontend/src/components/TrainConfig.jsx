import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  AlertTriangle, CheckCircle2, Cloud, Cpu, Database, Eye, EyeOff, KeyRound, Layers, Laptop, Rocket, Save, ShieldCheck,
  SlidersHorizontal, XCircle,
} from 'lucide-react'
import { api, timeAgo } from '../lib/api'
import { Badge, Button, Card, Field, Input, Segmented, Select, cx } from '../lib/ui'

const IMG_SIZES = [320, 416, 480, 512, 640, 736, 800, 960, 1024, 1280]

export default function TrainConfig({ project, catalog, onLaunched }) {
  const [cfg, setCfg] = useState(null)
  const [appCfg, setAppCfg] = useState(null)
  const [stats, setStats] = useState(null)
  const [launching, setLaunching] = useState(null)

  useEffect(() => {
    api.config().then((c) => {
      setAppCfg(c)
      setCfg(c.training_defaults)
    }).catch((e) => toast.error(e.message))
    api.stats(project.name).then(setStats).catch(() => {})
  }, [project.name])

  const family = catalog?.families.find((f) => f.id === cfg?.family)
  const modelName = useMemo(() => {
    if (!family || !cfg) return ''
    return `${family.prefix}${cfg.variant}.${cfg.pretrained ? 'pt' : 'yaml'}`
  }, [family, cfg])

  if (!cfg || !catalog || !appCfg) return <div className="p-8 text-sm text-slate-500">Loading…</div>

  const set = (patch) => setCfg((c) => ({ ...c, ...patch }))
  const setFamily = (id) => {
    const fam = catalog.families.find((f) => f.id === id)
    const variants = fam.variants.map((v) => v.id)
    set({ family: id, variant: variants.includes(cfg.variant) ? cfg.variant : variants[0] })
  }

  const ds = stats?.dataset
  const issues = []
  if (ds) {
    if (!ds.per_class.length) issues.push({ level: 'error', text: 'No classes defined.' })
    if (!ds.labeled_total) issues.push({ level: 'error', text: 'No labeled images yet.' })
    else if (ds.labeled_total < 20) issues.push({ level: 'warn', text: `Only ${ds.labeled_total} labeled images — expect weak results (aim for 100+ per class).` })
    if (ds.labeled_total && !ds.images.val) issues.push({ level: 'warn', text: 'Validation split is empty; training images will be reused for validation.' })
    ds.per_class.filter((c) => c.boxes === 0).forEach((c) => issues.push({ level: 'warn', text: `Class "${c.name}" has no boxes.` }))
  }
  const blocked = issues.some((i) => i.level === 'error')
  const kaggleReady = appCfg.kaggle.username && appCfg.kaggle.has_key

  const launch = async (backend) => {
    setLaunching(backend)
    try {
      const run = await api.createRun(project.name, backend, cfg)
      toast.success(backend === 'kaggle' ? 'Dataset packaging started — pushing to Kaggle' : 'Local training started')
      onLaunched(run.id)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setLaunching(null)
    }
  }

  const saveDefaults = async () => {
    try {
      await api.saveDefaults(cfg)
      toast.success('Saved as default training settings')
    } catch (e) {
      toast.error(e.message)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-[1fr_340px]">
        <div className="space-y-6">
          <header>
            <h1 className="text-xl font-semibold text-slate-50">Training setup</h1>
            <p className="text-sm text-slate-500">Choose an architecture and hyper-parameters, then train on Kaggle GPUs or this machine.</p>
          </header>

          <Card title="Model architecture" icon={Layers} bodyClass="space-y-5">
            <Field label="Model family">
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                {catalog.families.map((f) => (
                  <button key={f.id} onClick={() => setFamily(f.id)}
                          className={cx('rounded-lg border px-2 py-2.5 text-sm font-semibold transition-colors',
                            cfg.family === f.id ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300' : 'border-slate-800 bg-slate-950 text-slate-300 hover:border-slate-700')}>
                    {f.label}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Variant / size" hint="Larger variants are more accurate but slower and need more GPU memory.">
              <div className="flex flex-wrap gap-2">
                {family?.variants.map((v) => (
                  <button key={v.id} onClick={() => set({ variant: v.id })}
                          className={cx('min-w-24 rounded-lg border px-3 py-2 text-left transition-colors',
                            cfg.variant === v.id ? 'border-emerald-500 bg-emerald-500/10' : 'border-slate-800 bg-slate-950 hover:border-slate-700')}>
                    <span className={cx('block text-sm font-semibold', cfg.variant === v.id ? 'text-emerald-300' : 'text-slate-200')}>{v.label}</span>
                    <span className="block font-mono text-[11px] text-slate-500">{family.prefix}{v.id}</span>
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Initialization">
              <div className="grid gap-2 sm:grid-cols-2">
                {[
                  { v: true, title: 'Pre-trained (transfer learning)', desc: 'Start from official COCO weights. Best for most custom datasets.' },
                  { v: false, title: 'From scratch', desc: 'Build from YAML with random weights. Needs much more data and epochs.' },
                ].map((o) => (
                  <button key={String(o.v)} onClick={() => set({ pretrained: o.v })}
                          className={cx('rounded-lg border p-3 text-left transition-colors',
                            cfg.pretrained === o.v ? 'border-emerald-500 bg-emerald-500/10' : 'border-slate-800 bg-slate-950 hover:border-slate-700')}>
                    <span className="block text-sm font-medium text-slate-100">{o.title}</span>
                    <span className="mt-0.5 block text-xs text-slate-500">{o.desc}</span>
                  </button>
                ))}
              </div>
            </Field>
            <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-400">
              Checkpoint: <code className="font-mono text-emerald-400">YOLO("{modelName}")</code>
            </div>
          </Card>

          <Card title="Hyper-parameters" icon={SlidersHorizontal}
                actions={<Button size="sm" variant="ghost" icon={Save} onClick={saveDefaults}>Save as default</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Field label="Epochs">
                <Input type="number" min="1" value={cfg.epochs} onChange={(e) => set({ epochs: e.target.value })} />
              </Field>
              <Field label="Batch size" hint="-1 = auto (60% GPU memory)">
                <Input type="number" min="-1" value={cfg.batch} onChange={(e) => set({ batch: e.target.value })} />
              </Field>
              <Field label="Image size (imgsz)">
                <Select value={cfg.imgsz} onChange={(e) => set({ imgsz: Number(e.target.value) })}>
                  {IMG_SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
                </Select>
              </Field>
              <Field label="Optimizer">
                <Select value={cfg.optimizer} onChange={(e) => set({ optimizer: e.target.value })}>
                  {catalog.optimizers.map((o) => <option key={o} value={o}>{o === 'auto' ? 'Auto' : o}</option>)}
                </Select>
              </Field>
              <Field label="Initial learning rate (lr0)">
                <Input type="number" step="0.001" min="0" value={cfg.lr0} onChange={(e) => set({ lr0: e.target.value })} />
              </Field>
              <Field label="Early-stop patience" hint="Epochs without improvement">
                <Input type="number" min="0" value={cfg.patience} onChange={(e) => set({ patience: e.target.value })} />
              </Field>
            </div>
          </Card>

          <Card title="Compute" icon={Cpu} bodyClass="grid gap-4 sm:grid-cols-2">
            <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-950 p-3">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-200"><Cloud className="size-4 text-sky-400" /> Kaggle GPU</div>
              <Field label="Accelerator">
                <Select value={cfg.accelerator} onChange={(e) => set({ accelerator: e.target.value })}>
                  {catalog.accelerators.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
                </Select>
              </Field>
              <Field label="Device" hint="0,1 runs DDP on both T4s">
                <Segmented size="sm" value={cfg.kaggle_device} onChange={(v) => set({ kaggle_device: v })}
                           options={[{ value: '0', label: 'GPU 0' }, { value: '0,1', label: 'GPU 0,1' }]} />
              </Field>
            </div>
            <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-950 p-3">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-200"><Laptop className="size-4 text-slate-400" /> Local machine</div>
              <Field label="Device" hint="Auto picks CUDA GPU 0 when available, else CPU">
                <Segmented size="sm" value={cfg.local_device} onChange={(v) => set({ local_device: v })}
                           options={[{ value: 'auto', label: 'Auto' }, { value: '0', label: 'GPU 0' }, { value: 'cpu', label: 'CPU' }]} />
              </Field>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <KaggleCredentials appCfg={appCfg} onSaved={setAppCfg} />

          <Card title="Dataset readiness" icon={Database} bodyClass="space-y-3">
            {ds ? (
              <>
                <div className="grid grid-cols-3 gap-2 text-center">
                  {[['Train', ds.images.train], ['Val', ds.images.val], ['Boxes', ds.boxes_total]].map(([l, v]) => (
                    <div key={l} className="rounded-lg bg-slate-950 py-2 ring-1 ring-slate-800">
                      <div className="text-lg font-semibold tabular-nums text-slate-100">{v}</div>
                      <div className="text-[10px] text-slate-500">{l}</div>
                    </div>
                  ))}
                </div>
                <ul className="space-y-1.5">
                  {issues.length === 0 && (
                    <li className="flex items-center gap-2 text-xs text-emerald-400"><CheckCircle2 className="size-3.5" /> Dataset looks ready.</li>
                  )}
                  {issues.map((i, n) => (
                    <li key={n} className={cx('flex gap-2 text-xs', i.level === 'error' ? 'text-rose-400' : 'text-amber-400')}>
                      {i.level === 'error' ? <XCircle className="mt-0.5 size-3.5 shrink-0" /> : <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />}
                      {i.text}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="text-xs text-slate-500">Loading…</p>
            )}
          </Card>

          <div className="space-y-2">
            <Button variant="primary" size="lg" className="w-full" icon={Rocket} loading={launching === 'kaggle'}
                    disabled={blocked || !kaggleReady || launching} onClick={() => launch('kaggle')}>
              Launch on Kaggle GPU
            </Button>
            {!kaggleReady && <p className="text-center text-[11px] text-amber-400">Save Kaggle credentials to enable remote training.</p>}
            <Button size="lg" className="w-full" icon={Laptop} loading={launching === 'local'} disabled={blocked || launching}
                    onClick={() => launch('local')}>
              Train locally
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function KaggleCredentials({ appCfg, onSaved }) {
  const k = appCfg.kaggle
  const [username, setUsername] = useState(k.username || '')
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(null)
  const [result, setResult] = useState(null)

  const save = async () => {
    setBusy('save')
    try {
      const cfg = await api.saveKaggle(username, key)
      onSaved(cfg)
      setKey('')
      setResult(null)
      toast.success('Kaggle credentials saved to workspace/config.json')
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(null)
    }
  }

  const verify = async () => {
    setBusy('verify')
    try {
      // Verify the typed values when present, otherwise the stored ones.
      const res = await api.verifyKaggle(key ? { username, key } : {})
      setResult(res)
      if (res.ok && !key) onSaved(await api.config())
      res.ok ? toast.success(res.message) : toast.error(res.message)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(null)
    }
  }

  const onFile = async (file) => {
    try {
      const data = JSON.parse(await file.text())
      setUsername(data.username || '')
      setKey(data.key || '')
      toast('Loaded kaggle.json — click Save')
    } catch {
      toast.error('Not a valid kaggle.json')
    }
  }

  return (
    <Card title="Kaggle API" icon={KeyRound}
          actions={k.verified ? <Badge tone="emerald"><ShieldCheck className="size-3" /> Verified</Badge>
            : k.has_key ? <Badge tone="amber">Not verified</Badge> : <Badge>Not set</Badge>}
          bodyClass="space-y-3">
      <Field label="Username">
        <Input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="kaggle username" autoComplete="off" />
      </Field>
      <Field label="API key" hint={k.has_key ? `Stored key ${k.key_hint} — leave blank to keep it` : 'From kaggle.com → Settings → API → Create New Token'}>
        <div className="relative">
          <Input type={show ? 'text' : 'password'} value={key} onChange={(e) => setKey(e.target.value)}
                 placeholder={k.has_key ? '••••••••••••' : 'paste key'} autoComplete="off" className="pr-9" />
          <button onClick={() => setShow((s) => !s)} className="absolute right-2 top-2 text-slate-500 hover:text-slate-300" type="button"
                  title={show ? 'Hide' : 'Show'}>
            {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
      </Field>
      <label className="block cursor-pointer text-[11px] text-emerald-400 hover:underline">
        Import kaggle.json…
        <input type="file" accept=".json" hidden onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} />
      </label>
      <div className="flex gap-2">
        <Button className="flex-1" icon={Save} loading={busy === 'save'} disabled={!username.trim() || (!key && !k.has_key)} onClick={save}>Save</Button>
        <Button className="flex-1" variant="ghost" icon={ShieldCheck} loading={busy === 'verify'} disabled={!k.has_key && !key} onClick={verify}>Verify</Button>
      </div>
      {result && (
        <p className={cx('text-xs', result.ok ? 'text-emerald-400' : 'text-rose-400')}>{result.message}</p>
      )}
      {k.verified_at && <p className="text-[11px] text-slate-500">Last verified {timeAgo(k.verified_at)}</p>}
      <p className="text-[11px] leading-relaxed text-slate-500">
        Your Kaggle account must be phone-verified to use GPUs and internet access in kernels.
      </p>
    </Card>
  )
}
