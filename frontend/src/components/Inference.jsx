import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Boxes, Brain, Download, Film, ImagePlus, Loader2, Play, Plug, Radio, Scan, Square, Trash2, Upload, UploadCloud,
} from 'lucide-react'
import { api, formatBytes, formatDuration, timeAgo, uploadModel } from '../lib/api'
import { Badge, Button, Card, EmptyState, Field, Input, Modal, Progress, Segmented, Select, cx, confirmAction } from '../lib/ui'

const IMG_SIZES = [0, 320, 416, 480, 512, 640, 800, 960, 1024, 1280]
const FALLBACK_COLORS = ['#10B981', '#F59E0B', '#3B82F6', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316']

const loadPref = (key, fallback) => {
  try {
    const v = localStorage.getItem(key)
    return v ? { ...fallback, ...JSON.parse(v) } : fallback
  } catch {
    return fallback
  }
}

export default function Inference({ project, catalog }) {
  const pname = project.name
  const prefKey = `yf.infer.${pname}`
  const [models, setModels] = useState([])
  const [opts, setOpts] = useState(() => loadPref(prefKey, { model: '', conf: 0.25, iou: 0.5, imgsz: 0, classes: [] }))
  const [modelClasses, setModelClasses] = useState([])
  const [source, setSource] = useState('image')
  const [uploadOpen, setUploadOpen] = useState(false)
  const colors = catalog?.class_colors || FALLBACK_COLORS

  const loadModels = useCallback(async () => {
    try {
      const list = await api.models(pname)
      setModels(list)
      setOpts((o) => (list.some((m) => m.key === o.model) ? o : { ...o, model: list[0]?.key || '', classes: [] }))
      return list
    } catch (e) {
      toast.error(e.message)
      return []
    }
  }, [pname])

  useEffect(() => {
    loadModels()
  }, [loadModels])

  useEffect(() => {
    try {
      localStorage.setItem(prefKey, JSON.stringify(opts))
    } catch {
      /* ignore */
    }
  }, [opts, prefKey])

  useEffect(() => {
    if (!opts.model) return
    const m = models.find((x) => x.key === opts.model)
    if (m?.classes?.length) {
      setModelClasses(m.classes)
      return
    }
    api.autolabelClasses(pname, opts.model).then(setModelClasses).catch(() => setModelClasses([]))
  }, [opts.model, models, pname])

  const set = (patch) => setOpts((o) => ({ ...o, ...patch }))
  const selected = models.find((m) => m.key === opts.model)
  const groups = [
    ['Trained in this project', models.filter((m) => m.kind === 'run')],
    ['Uploaded models', models.filter((m) => m.kind === 'custom')],
    ['COCO pre-trained', models.filter((m) => m.kind === 'base')],
  ]
  const custom = groups[1][1]

  const removeModel = async (m) => {
    if (!confirmAction(`Delete uploaded model "${m.label}"?`)) return
    try {
      await api.deleteModel(m.key.slice('custom:'.length))
      toast.success('Model deleted')
      loadModels()
    } catch (e) {
      toast.error(e.message)
    }
  }

  return (
    <div className="flex h-full">
      {/* ------------------------------------------------ model & settings */}
      <aside className="w-80 shrink-0 space-y-4 overflow-y-auto border-r border-slate-800 bg-slate-950 p-3">
        <Card title="Model" icon={Brain} bodyClass="space-y-3"
              actions={<Button size="sm" variant="ghost" icon={Upload} onClick={() => setUploadOpen(true)}>Upload</Button>}>
          <Select value={opts.model} onChange={(e) => set({ model: e.target.value, classes: [] })}>
            {groups.map(([label, list]) => list.length > 0 && (
              <optgroup key={label} label={label}>
                {list.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </optgroup>
            ))}
          </Select>
          {selected && (
            <div className="flex flex-wrap gap-1.5 text-[11px]">
              <Badge tone={selected.kind === 'run' ? 'emerald' : selected.kind === 'custom' ? 'sky' : 'slate'}>
                {selected.kind === 'run' ? 'Project run' : selected.kind === 'custom' ? `Uploaded · ${selected.format}` : 'COCO'}
              </Badge>
              <Badge>{modelClasses.length} classes</Badge>
            </div>
          )}
          {custom.length > 0 && (
            <ul className="space-y-1 border-t border-slate-800 pt-3">
              <li className="text-[11px] font-medium text-slate-500">Uploaded models</li>
              {custom.map((m) => (
                <li key={m.key} className="group flex items-center gap-2 rounded-md px-1.5 py-1 hover:bg-slate-900">
                  <Boxes className="size-3.5 shrink-0 text-sky-400" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-slate-200">{m.label}</span>
                    <span className="block text-[10px] text-slate-500">{m.format} · {formatBytes(m.size)} · {m.classes?.length} classes · {timeAgo(m.uploaded_at)}</span>
                  </span>
                  <button onClick={() => removeModel(m)} title="Delete model"
                          className="rounded p-1 text-slate-500 opacity-0 hover:bg-rose-600/20 hover:text-rose-400 group-hover:opacity-100">
                    <Trash2 className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Detection settings" icon={Scan} bodyClass="space-y-4">
          <Slider label="Confidence threshold" value={opts.conf} min={0.05} max={0.95} onChange={(conf) => set({ conf })} />
          <Slider label="IoU (NMS) threshold" value={opts.iou} min={0.1} max={0.9} onChange={(iou) => set({ iou })} />
          <Field label="Inference size">
            <Select value={opts.imgsz} onChange={(e) => set({ imgsz: Number(e.target.value) })}>
              {IMG_SIZES.map((s) => <option key={s} value={s}>{s ? `${s}px` : 'Model default'}</option>)}
            </Select>
          </Field>
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
              <span>Classes</span>
              {opts.classes.length > 0 && <button className="text-emerald-400 hover:underline" onClick={() => set({ classes: [] })}>Show all</button>}
            </div>
            <div className="flex max-h-40 flex-wrap gap-1 overflow-y-auto">
              {modelClasses.map((c, i) => {
                const on = opts.classes.includes(c)
                return (
                  <button key={c} onClick={() => set({ classes: on ? opts.classes.filter((x) => x !== c) : [...opts.classes, c] })}
                          className={cx('flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] transition-colors',
                            on ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-300' : 'border-slate-800 text-slate-400 hover:border-slate-700')}>
                    <span className="size-1.5 rounded-full" style={{ background: colors[i % colors.length] }} />
                    {c}
                  </button>
                )
              })}
            </div>
            <p className="mt-1 text-[10px] text-slate-500">{opts.classes.length ? `Detecting ${opts.classes.length} selected classes` : 'Detecting all classes'}</p>
          </div>
        </Card>
      </aside>

      {/* ------------------------------------------------ source & results */}
      <section className="min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl space-y-5 p-6">
          <header className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-xl font-semibold text-slate-50">Test model</h1>
              <p className="text-sm text-slate-500">Run a trained, uploaded or COCO YOLO model on images, videos or a live stream.</p>
            </div>
            <Segmented value={source} onChange={setSource}
                       options={[{ value: 'image', label: 'Image' }, { value: 'video', label: 'Video' }, { value: 'live', label: 'Live RTSP' }]} />
          </header>
          {!opts.model ? (
            <EmptyState icon={Brain} title="No model selected">Upload a YOLO model or train one first.</EmptyState>
          ) : source === 'image' ? (
            <ImageTest project={project} opts={opts} colors={colors} modelClasses={modelClasses} />
          ) : source === 'video' ? (
            <VideoTest project={project} opts={opts} />
          ) : (
            <LiveTest project={project} opts={opts} />
          )}
        </div>
      </section>

      <UploadModelModal open={uploadOpen} onClose={() => setUploadOpen(false)} onUploaded={async (m) => {
        await loadModels()
        set({ model: m.key, classes: [] })
        setUploadOpen(false)
      }} />
    </div>
  )
}

function Slider({ label, value, min, max, onChange }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span className="tabular-nums text-slate-200">{value.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step="0.05" value={value} className="w-full" onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

const optsForm = (opts) => {
  const form = new FormData()
  form.append('model', opts.model)
  form.append('conf', String(opts.conf))
  form.append('iou', String(opts.iou))
  form.append('imgsz', String(opts.imgsz || 0))
  form.append('classes', JSON.stringify(opts.classes || []))
  return form
}

// ------------------------------------------------------------------ model upload

function UploadModelModal({ open, onClose, onUploaded }) {
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [progress, setProgress] = useState(null)

  useEffect(() => {
    if (open) {
      setFile(null)
      setName('')
      setProgress(null)
    }
  }, [open])

  const pick = (f) => {
    if (!f) return
    if (!/\.(pt|onnx)$/i.test(f.name)) {
      toast.error('Only YOLO model files (.pt or .onnx) are accepted')
      return
    }
    setFile(f)
    if (!name) setName(f.name.replace(/\.(pt|onnx)$/i, ''))
  }

  const submit = async () => {
    setProgress(0)
    try {
      const m = await uploadModel(file, name, setProgress)
      toast.success(`Model "${m.name}" added · ${m.classes.length} classes`)
      onUploaded(m)
    } catch (e) {
      toast.error(e.message)
      setProgress(null)
    }
  }

  const validating = progress !== null && progress >= 1
  return (
    <Modal open={open} onClose={progress === null ? onClose : () => {}} title="Upload YOLO model"
           footer={
             <>
               <Button variant="ghost" onClick={onClose} disabled={progress !== null}>Cancel</Button>
               <Button variant="primary" icon={Upload} disabled={!file} loading={progress !== null} onClick={submit}>
                 {validating ? 'Validating…' : 'Upload'}
               </Button>
             </>
           }>
      <div className="space-y-4">
        <label
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            pick(e.dataTransfer.files[0])
          }}
          className="flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed border-slate-800 bg-slate-950 px-6 py-8 text-center hover:border-slate-700"
        >
          <input type="file" accept=".pt,.onnx" hidden onChange={(e) => pick(e.target.files[0])} />
          <UploadCloud className="mb-2 size-7 text-slate-400" />
          {file ? (
            <span className="text-sm text-slate-200">{file.name} <span className="text-slate-500">· {formatBytes(file.size)}</span></span>
          ) : (
            <span className="text-sm text-slate-300">Drop a .pt or .onnx file, or click to browse</span>
          )}
        </label>
        <Field label="Display name">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. PPE detector v2" />
        </Field>
        {progress !== null && (
          <div className="space-y-1.5">
            <Progress value={progress} />
            <p className="text-xs text-slate-400">{validating ? 'Checking it is an Ultralytics YOLO detection model…' : `Uploading ${Math.round(progress * 100)}%`}</p>
          </div>
        )}
        <ul className="space-y-1 text-[11px] text-slate-500">
          <li>• Only <span className="text-slate-300">Ultralytics YOLO detection</span> models (v5u, v8–v12, YOLO26) are accepted. Each one is test-loaded before it's saved.</li>
          <li>• ONNX files must be exported with Ultralytics so class names are embedded.</li>
          <li>• .pt files can run code when loaded, so only upload models from sources you trust.</li>
        </ul>
      </div>
    </Modal>
  )
}

// ------------------------------------------------------------------ image

function ImageTest({ project, opts, colors, modelClasses }) {
  const [src, setSrc] = useState(null) // { kind: 'file', file, url } | { kind: 'project', id, url }
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [hover, setHover] = useState(null)
  const [images, setImages] = useState([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const inputRef = useRef(null)
  const reqRef = useRef(0)

  useEffect(() => () => src?.kind === 'file' && URL.revokeObjectURL(src.url), [src])

  const run = useCallback(async () => {
    if (!src) return
    const token = ++reqRef.current
    setBusy(true)
    const form = optsForm(opts)
    if (src.kind === 'file') form.append('file', src.file)
    else form.append('image_id', src.id)
    try {
      const res = await api.inferImage(project.name, form)
      if (token === reqRef.current) setResult(res)
    } catch (e) {
      if (token === reqRef.current) toast.error(e.message)
    } finally {
      if (token === reqRef.current) setBusy(false)
    }
  }, [src, opts, project.name])

  // Re-run automatically when the image or any setting changes.
  useEffect(() => {
    const t = setTimeout(run, 250)
    return () => clearTimeout(t)
  }, [run])

  const pickFile = (f) => {
    if (!f) return
    if (!/\.(jpe?g|png|webp|bmp)$/i.test(f.name)) {
      toast.error('Choose a JPG, PNG, WebP or BMP image')
      return
    }
    setResult(null)
    setSrc({ kind: 'file', file: f, url: URL.createObjectURL(f), name: f.name })
  }

  const colorFor = (d) => {
    const i = modelClasses.indexOf(d.class_name)
    return colors[(i >= 0 ? i : d.cls) % colors.length]
  }

  const downloadAnnotated = async () => {
    const img = new Image()
    img.src = src.url
    await img.decode()
    const c = document.createElement('canvas')
    c.width = img.naturalWidth
    c.height = img.naturalHeight
    const ctx = c.getContext('2d')
    ctx.drawImage(img, 0, 0)
    const lw = Math.max(2, Math.round(c.width / 400))
    ctx.font = `600 ${Math.max(12, lw * 6)}px Inter, sans-serif`
    for (const d of result.detections) {
      const color = colorFor(d)
      const x = d.x1 * c.width
      const y = d.y1 * c.height
      ctx.lineWidth = lw
      ctx.strokeStyle = color
      ctx.strokeRect(x, y, (d.x2 - d.x1) * c.width, (d.y2 - d.y1) * c.height)
      const label = `${d.class_name} ${(d.conf * 100).toFixed(0)}%`
      const th = Math.max(12, lw * 6) + 6
      ctx.fillStyle = color
      ctx.fillRect(x, Math.max(0, y - th), ctx.measureText(label).width + 8, th)
      ctx.fillStyle = '#020617'
      ctx.fillText(label, x + 4, Math.max(0, y - th) + th - 6)
    }
    const a = document.createElement('a')
    a.href = c.toDataURL('image/png')
    a.download = `${(src.name || 'image').replace(/\.[^.]+$/, '')}_detections.png`
    a.click()
  }

  if (!src) {
    return (
      <div className="space-y-3">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            pickFile(e.dataTransfer.files[0])
          }}
          onClick={() => inputRef.current?.click()}
          className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-800 bg-slate-900/50 px-6 py-16 text-center hover:border-slate-700"
        >
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" hidden onChange={(e) => pickFile(e.target.files[0])} />
          <ImagePlus className="mb-3 size-8 text-slate-400" />
          <p className="text-sm font-medium text-slate-200">Drop an image to test, or click to browse</p>
          <p className="mt-1 text-xs text-slate-500">The image is only used for this test; it isn't added to the project.</p>
        </div>
        <div className="text-center">
          <Button variant="ghost" size="sm" onClick={async () => {
            setImages(await api.images(project.name))
            setPickerOpen(true)
          }}>…or pick an image from this project</Button>
        </div>
        <ProjectImagePicker open={pickerOpen} project={project} images={images} onClose={() => setPickerOpen(false)}
                            onPick={(im) => {
                              setPickerOpen(false)
                              setResult(null)
                              setSrc({ kind: 'project', id: im.id, url: api.fileUrl(project.name, im.id), name: im.filename })
                            }} />
      </div>
    )
  }

  const dets = result?.detections || []
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <div className="space-y-3">
        <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black">
          <img src={src.url} alt="" className="block w-full" />
          {result && (
            <svg viewBox={`0 0 ${result.width} ${result.height}`} preserveAspectRatio="none" className="absolute inset-0 size-full">
              {dets.map((d, i) => {
                const color = colorFor(d)
                const x = d.x1 * result.width
                const y = d.y1 * result.height
                const w = (d.x2 - d.x1) * result.width
                const h = (d.y2 - d.y1) * result.height
                const lw = Math.max(2, result.width / 400)
                const fs = Math.max(11, result.width / 60)
                const label = `${d.class_name} ${(d.conf * 100).toFixed(0)}%`
                const dim = hover !== null && hover !== i
                return (
                  <g key={i} opacity={dim ? 0.25 : 1} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                    <rect x={x} y={y} width={w} height={h} fill={color} fillOpacity={hover === i ? 0.2 : 0.08} stroke={color} strokeWidth={lw} />
                    <rect x={x - lw / 2} y={Math.max(0, y - fs * 1.5)} width={label.length * fs * 0.58 + fs * 0.6} height={fs * 1.5} fill={color} />
                    <text x={x + fs * 0.3} y={Math.max(0, y - fs * 1.5) + fs * 1.1} fontSize={fs} fontWeight="600" fill="#020617">{label}</text>
                  </g>
                )
              })}
            </svg>
          )}
          {busy && (
            <div className="absolute right-3 top-3 flex items-center gap-2 rounded-md bg-slate-950/80 px-2.5 py-1 text-xs text-amber-300">
              <Loader2 className="size-3.5 animate-spin" /> Detecting…
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" icon={ImagePlus} onClick={() => inputRef.current?.click()}>Another image</Button>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" hidden onChange={(e) => pickFile(e.target.files[0])} />
          <Button size="sm" variant="ghost" onClick={async () => {
            setImages(await api.images(project.name))
            setPickerOpen(true)
          }}>Pick project image</Button>
          {result && <Button size="sm" variant="ghost" icon={Download} onClick={downloadAnnotated}>Download annotated PNG</Button>}
        </div>
      </div>

      <div className="space-y-4">
        <Card title={`Detections (${dets.length})`} icon={Scan}
              actions={result && <span className="text-[11px] tabular-nums text-slate-500">{result.inference_ms} ms · {result.width}×{result.height}</span>}>
          {!result ? (
            <p className="text-xs text-slate-500">Running…</p>
          ) : !dets.length ? (
            <p className="text-xs text-slate-500">Nothing detected above {opts.conf.toFixed(2)} confidence. Try lowering the threshold.</p>
          ) : (
            <div className="space-y-4">
              <ul className="space-y-1.5">
                {Object.entries(result.counts).map(([name, n]) => (
                  <li key={name} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2 text-slate-200">
                      <span className="size-2.5 rounded-sm" style={{ background: colorFor({ class_name: name, cls: 0 }) }} />
                      {name}
                    </span>
                    <span className="tabular-nums text-slate-300">{n}</span>
                  </li>
                ))}
              </ul>
              <div className="max-h-80 overflow-y-auto rounded-lg border border-slate-800">
                <table className="w-full text-[11px]">
                  <thead className="sticky top-0 bg-slate-900 text-left text-slate-400">
                    <tr><th className="px-2 py-1.5 font-medium">#</th><th className="px-2 py-1.5 font-medium">Class</th><th className="px-2 py-1.5 text-right font-medium">Conf</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {dets.map((d, i) => (
                      <tr key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                          className={cx('cursor-default', hover === i ? 'bg-slate-800' : 'hover:bg-slate-900')}>
                        <td className="px-2 py-1.5 tabular-nums text-slate-500">{i + 1}</td>
                        <td className="px-2 py-1.5 text-slate-200">{d.class_name}</td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">{(d.conf * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Card>
      </div>
      <ProjectImagePicker open={pickerOpen} project={project} images={images} onClose={() => setPickerOpen(false)}
                          onPick={(im) => {
                            setPickerOpen(false)
                            setResult(null)
                            setSrc({ kind: 'project', id: im.id, url: api.fileUrl(project.name, im.id), name: im.filename })
                          }} />
    </div>
  )
}

function ProjectImagePicker({ open, project, images, onClose, onPick }) {
  const [q, setQ] = useState('')
  const shown = useMemo(() => images.filter((i) => i.filename.toLowerCase().includes(q.toLowerCase())).slice(0, 300), [images, q])
  return (
    <Modal open={open} onClose={onClose} title="Pick a project image" width="max-w-3xl">
      <Input autoFocus placeholder="Filter by file name…" value={q} onChange={(e) => setQ(e.target.value)} className="mb-3" />
      {!images.length ? (
        <p className="text-sm text-slate-500">This project has no images yet.</p>
      ) : (
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
          {shown.map((im) => (
            <button key={im.id} onClick={() => onPick(im)} className="overflow-hidden rounded-lg border border-slate-800 text-left hover:border-emerald-500">
              <img src={api.thumbUrl(project.name, im.id)} loading="lazy" alt="" className="aspect-video w-full bg-slate-900 object-cover" />
              <span className="block truncate px-1.5 py-1 text-[10px] text-slate-400">{im.filename}</span>
            </button>
          ))}
        </div>
      )}
    </Modal>
  )
}

// ------------------------------------------------------------------ video

function VideoTest({ project, opts }) {
  const [file, setFile] = useState(null)
  const [mediaFile, setMediaFile] = useState('')
  const [videos, setVideos] = useState([])
  const [job, setJob] = useState(null)
  const [results, setResults] = useState([])
  const [playing, setPlaying] = useState(null)
  const esRef = useRef(null)
  const inputRef = useRef(null)

  const loadResults = useCallback(() => api.videoResults(project.name).then(setResults).catch(() => {}), [project.name])

  useEffect(() => {
    loadResults()
    api.media(project.name).then((m) => setVideos(m.filter((x) => x.type === 'video'))).catch(() => {})
    return () => esRef.current?.close()
  }, [project.name, loadResults])

  const start = async () => {
    const form = optsForm(opts)
    if (file) form.append('file', file)
    else form.append('media_filename', mediaFile)
    try {
      const j = await api.inferVideo(project.name, form)
      setJob(j)
      const es = new EventSource(api.jobEventsUrl(j.id))
      esRef.current = es
      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data)
        setJob(data)
        if (data.status !== 'running') {
          es.close()
          if (data.status === 'completed') {
            toast.success('Video processed')
            setPlaying(data.result)
            setFile(null)
          } else if (data.status === 'failed') toast.error(data.error || 'Processing failed')
          loadResults()
        }
      }
      es.onerror = () => es.close()
    } catch (e) {
      toast.error(e.message)
    }
  }

  const remove = async (r) => {
    if (!confirmAction('Delete this processed video?')) return
    await api.deleteVideoResult(project.name, r.id)
    if (playing?.id === r.id) setPlaying(null)
    loadResults()
  }

  const running = job?.status === 'running'
  return (
    <div className="space-y-5">
      <Card title="Process a video" icon={Film} bodyClass="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              const f = e.dataTransfer.files[0]
              if (f) {
                setFile(f)
                setMediaFile('')
              }
            }}
            onClick={() => !running && inputRef.current?.click()}
            className={cx('flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center',
              file ? 'border-emerald-500/60 bg-emerald-500/5' : 'border-slate-800 bg-slate-950 hover:border-slate-700')}
          >
            <input ref={inputRef} type="file" accept=".mp4,.avi,.mkv,.mov,.webm,.m4v" hidden
                   onChange={(e) => {
                     if (e.target.files[0]) {
                       setFile(e.target.files[0])
                       setMediaFile('')
                     }
                     e.target.value = ''
                   }} />
            <UploadCloud className="mb-2 size-6 text-slate-400" />
            <span className="text-sm text-slate-200">{file ? `${file.name} · ${formatBytes(file.size)}` : 'Upload a video file'}</span>
            <span className="mt-1 text-[11px] text-slate-500">MP4, AVI, MKV, MOV, WebM (not added to project media)</span>
          </div>
          <Field label="…or choose a project video" hint={videos.length ? 'Uploads and RTSP recordings from Media & Capture' : 'No videos in this project yet'}>
            <Select value={mediaFile} disabled={!videos.length} onChange={(e) => {
              setMediaFile(e.target.value)
              setFile(null)
            }}>
              <option value="">Select video…</option>
              {videos.map((v) => (
                <option key={v.filename} value={v.filename}>
                  {v.filename}{v.info?.duration ? ` (${formatDuration(v.info.duration)})` : ''}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        {running || job?.status === 'failed' ? (
          <div className="space-y-2">
            <Progress value={job.progress} tone={job.status === 'failed' ? 'rose' : 'amber'} />
            <div className="flex justify-between text-xs">
              <span className={job.status === 'failed' ? 'text-rose-400' : 'text-slate-300'}>{job.status === 'failed' ? job.error : job.message}</span>
              {running && <Button size="sm" variant="dangerGhost" icon={Square} onClick={() => api.cancelJob(job.id)}>Cancel</Button>}
            </div>
          </div>
        ) : null}
        <Button variant="primary" icon={Play} disabled={(!file && !mediaFile) || running} loading={running} onClick={start}>
          Run detection on video
        </Button>
      </Card>

      {playing && (
        <Card title={`Result · ${playing.source}`} icon={Play}
              actions={<a href={api.videoResultUrl(project.name, playing.id)} download><Button size="sm" icon={Download}>Download</Button></a>}>
          {playing.browser_playable ? (
            <video key={playing.id} src={api.videoResultUrl(project.name, playing.id)} controls autoPlay muted className="w-full rounded-lg bg-black" />
          ) : (
            <p className="text-sm text-slate-400">This OpenCV build could only write MPEG-4 Part 2, which browsers can't play. Download it to watch.</p>
          )}
          <VideoSummary r={playing} />
        </Card>
      )}

      <Card title={`Processed videos (${results.length})`} icon={Film}>
        {!results.length ? (
          <p className="text-xs text-slate-500">No processed videos yet.</p>
        ) : (
          <ul className="divide-y divide-slate-800">
            {results.map((r) => (
              <li key={r.id} className="flex flex-wrap items-center gap-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-slate-200">{r.source}</div>
                  <div className="text-[11px] text-slate-500">
                    {r.model} · conf {r.conf} · {r.frames} frames · {Object.values(r.detections_total).reduce((a, b) => a + b, 0)} detections · {timeAgo(r.created_at)}
                  </div>
                </div>
                <Button size="sm" icon={Play} onClick={() => setPlaying(r)}>View</Button>
                <a href={api.videoResultUrl(project.name, r.id)} download><Button size="sm" variant="ghost" icon={Download} /></a>
                <Button size="sm" variant="dangerGhost" icon={Trash2} onClick={() => remove(r)} />
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

function VideoSummary({ r }) {
  const entries = Object.entries(r.detections_total)
  return (
    <div className="mt-4 grid gap-4 sm:grid-cols-[220px_1fr]">
      <dl className="space-y-1 text-xs">
        {[
          ['Frames', r.frames],
          ['Frames with detections', `${r.frames_with_detections} (${r.frames ? Math.round((r.frames_with_detections / r.frames) * 100) : 0}%)`],
          ['Processing time', formatDuration(r.processing_seconds)],
          ['Speed', `${r.processing_seconds ? (r.frames / r.processing_seconds).toFixed(1) : '—'} fps`],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2"><dt className="text-slate-500">{k}</dt><dd className="tabular-nums text-slate-300">{v}</dd></div>
        ))}
      </dl>
      {entries.length ? (
        <table className="w-full text-xs">
          <thead className="text-left text-slate-500">
            <tr><th className="pb-1 font-medium">Class</th><th className="pb-1 text-right font-medium">Total detections</th><th className="pb-1 text-right font-medium">Max in one frame</th></tr>
          </thead>
          <tbody className="tabular-nums text-slate-300">
            {entries.map(([name, n]) => (
              <tr key={name}><td className="py-0.5">{name}</td><td className="py-0.5 text-right">{n}</td><td className="py-0.5 text-right">{r.peak_per_frame[name]}</td></tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-xs text-slate-500">No objects detected.</p>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ live RTSP

function LiveTest({ project, opts }) {
  const [url, setUrl] = useState('')
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.liveSessions(project.name).then((l) => l[0] && setSession(l[0])).catch(() => {})
  }, [project.name])

  useEffect(() => {
    if (!session?.id) return
    const t = setInterval(() => api.liveStatus(session.id).then(setSession).catch(() => setSession(null)), 1000)
    return () => clearInterval(t)
  }, [session?.id])

  // Push threshold / class changes to the running session without restarting it.
  useEffect(() => {
    if (!session?.id) return
    const t = setTimeout(() => api.liveUpdate(session.id, { conf: opts.conf, iou: opts.iou, classes: opts.classes }).catch(() => {}), 300)
    return () => clearTimeout(t)
  }, [opts.conf, opts.iou, opts.classes, session?.id])

  const start = async () => {
    setBusy(true)
    try {
      setSession(await api.liveStart(project.name, {
        url, model: opts.model, conf: opts.conf, iou: opts.iou, imgsz: opts.imgsz || null, classes: opts.classes,
      }))
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    try {
      await api.liveStop(session.id)
    } catch {
      /* already gone */
    }
    setSession(null)
  }

  if (!session) {
    return (
      <Card title="Live detection" icon={Radio}>
        <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
          <Field label="Stream URL" hint="rtsp://user:pass@host:554/stream, or a webcam index like 0">
            <Input placeholder="rtsp://192.168.1.10:554/stream1" value={url} onChange={(e) => setUrl(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && url.trim() && start()} />
          </Field>
          <Button variant="primary" icon={Plug} loading={busy} disabled={!url.trim()} onClick={start} className="mb-5">Start detection</Button>
        </div>
        <p className="text-[11px] text-slate-500">The model runs on the newest frame each time, so a slow model drops frames instead of falling behind.</p>
      </Card>
    )
  }

  const live = session.status === 'live'
  const counts = Object.entries(session.counts || {})
  const modelChanged = session.model !== opts.model
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_280px]">
      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black">
        <img src={api.liveMjpegUrl(session.id)} alt="Live detections" className="aspect-video w-full object-contain" />
        {!live && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/70 text-sm text-slate-300">
            <Loader2 className="size-6 animate-spin text-amber-400" />
            {session.stream_error || 'Connecting…'}
          </div>
        )}
      </div>
      <div className="space-y-4">
        <Card title="Live stats" icon={Radio} actions={<Badge tone={live ? 'emerald' : 'amber'} dot pulse={!live}>{session.status}</Badge>}>
          <dl className="space-y-1.5 text-xs">
            {[
              ['Model', session.model],
              ['Resolution', session.width ? `${session.width}×${session.height}` : '—'],
              ['Inference FPS', session.infer_fps || '—'],
              ['Objects in frame', session.detections],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3">
                <dt className="text-slate-500">{k}</dt>
                <dd className="truncate text-right tabular-nums text-slate-300" title={String(v)}>{v}</dd>
              </div>
            ))}
          </dl>
          {counts.length > 0 && (
            <ul className="mt-3 space-y-1 border-t border-slate-800 pt-3">
              {counts.map(([name, n]) => (
                <li key={name} className="flex justify-between text-sm">
                  <span className="text-slate-200">{name}</span>
                  <span className="tabular-nums text-slate-300">{n}</span>
                </li>
              ))}
            </ul>
          )}
          {session.error && <p className="mt-3 text-xs text-rose-400">{session.error}</p>}
        </Card>
        {modelChanged && (
          <p className="text-[11px] text-amber-400">You picked a different model. Stop and start again to switch to it.</p>
        )}
        <p className="text-[11px] text-slate-500">The confidence, IoU and class settings apply straight away.</p>
        <Button className="w-full" variant="dangerGhost" icon={Square} onClick={stop}>Stop detection</Button>
      </div>
    </div>
  )
}
