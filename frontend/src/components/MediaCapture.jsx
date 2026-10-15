import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Camera, Circle, Film, ImagePlus, Images, Link2, Loader2, PenSquare, Plug, Radio, Scissors, Square, Trash2,
  Unplug, UploadCloud, Video,
} from 'lucide-react'
import { api, formatBytes, formatDuration, timeAgo, uploadFiles } from '../lib/api'
import { Badge, Button, Card, EmptyState, Field, Input, Modal, Progress, Segmented, cx, confirmAction } from '../lib/ui'

const ACCEPT = {
  photos: '.jpg,.jpeg,.png,.webp,.bmp',
  video: '.mp4,.avi,.mkv,.mov,.webm,.m4v',
}

export default function MediaCapture({ project, onChanged, onAnnotate }) {
  const [mode, setMode] = useState('photos')
  const [media, setMedia] = useState([])
  const [extracting, setExtracting] = useState(null)
  const [filter, setFilter] = useState('all')

  const load = useCallback(async () => {
    try {
      setMedia(await api.media(project.name))
    } catch (e) {
      toast.error(e.message)
    }
  }, [project.name])

  useEffect(() => {
    load()
  }, [load])

  const refresh = () => {
    load()
    onChanged?.()
  }

  const remove = async (m) => {
    if (!confirmAction(`Delete ${m.filename}?${m.type === 'image' ? ' Its annotation will be removed too.' : ''}`)) return
    try {
      await api.deleteMedia(project.name, m.filename)
      toast.success('Deleted')
      refresh()
    } catch (e) {
      toast.error(e.message)
    }
  }

  const shown = media.filter((m) => filter === 'all' || (filter === 'image' ? m.type === 'image' : m.type === 'video'))

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-slate-50">Media & Capture</h1>
            <p className="text-sm text-slate-500">Upload photos or videos, or record from a live RTSP camera.</p>
          </div>
          <Segmented
            value={mode}
            onChange={setMode}
            options={[
              { value: 'photos', label: 'Photos' },
              { value: 'video', label: 'Video file' },
              { value: 'rtsp', label: 'Live RTSP' },
            ]}
          />
        </header>

        {mode === 'rtsp' ? (
          <RtspPanel project={project} onRecorded={(entry) => {
            refresh()
            setExtracting(entry)
          }} onSnapshot={refresh} />
        ) : (
          <Dropzone key={mode} project={project} kind={mode} onUploaded={(saved) => {
            refresh()
            const firstVideo = saved.find((s) => s.type === 'video')
            if (firstVideo && saved.length === 1) setExtracting(firstVideo)
          }} />
        )}

        <Card
          title={`Media library (${media.length})`}
          icon={Images}
          actions={
            <>
              <Segmented size="sm" value={filter} onChange={setFilter}
                         options={[{ value: 'all', label: 'All' }, { value: 'image', label: 'Photos' }, { value: 'video', label: 'Videos' }]} />
              <Button size="sm" variant="primary" icon={PenSquare} onClick={onAnnotate}>Annotate</Button>
            </>
          }
        >
          {shown.length === 0 ? (
            <EmptyState icon={Film} title="Nothing here yet">Uploaded and recorded media will appear here.</EmptyState>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {shown.map((m) => (
                <MediaCard key={m.filename} project={project} m={m} onExtract={() => setExtracting(m)} onDelete={() => remove(m)} />
              ))}
            </div>
          )}
        </Card>
      </div>

      <ExtractModal project={project} media={extracting} onClose={() => setExtracting(null)} onDone={refresh} />
    </div>
  )
}

function MediaCard({ project, m, onExtract, onDelete }) {
  const isVideo = m.type === 'video'
  const info = m.info || {}
  return (
    <div className="group overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
      <div className="relative aspect-video bg-slate-900">
        <img src={api.thumbUrl(project.name, `raw_media/${m.filename}`)} alt="" loading="lazy" className="size-full object-cover" />
        <div className="absolute left-1.5 top-1.5 flex gap-1">
          <Badge tone={m.source === 'rtsp' ? 'amber' : isVideo ? 'sky' : 'slate'}>
            {m.source === 'rtsp' ? 'RTSP' : isVideo ? 'Video' : 'Photo'}
          </Badge>
        </div>
        {isVideo && info.duration != null && (
          <span className="absolute bottom-1.5 right-1.5 rounded bg-slate-950/80 px-1.5 py-0.5 text-[10px] tabular-nums text-slate-200">
            {formatDuration(info.duration)}
          </span>
        )}
        <button onClick={onDelete} title="Delete"
                className="absolute right-1.5 top-1.5 rounded-md bg-slate-950/80 p-1.5 text-slate-400 opacity-0 transition hover:bg-rose-600 hover:text-white group-hover:opacity-100">
          <Trash2 className="size-3.5" />
        </button>
      </div>
      <div className="space-y-2 p-2.5">
        <div className="truncate text-xs font-medium text-slate-200" title={m.filename}>{m.filename}</div>
        <div className="text-[11px] text-slate-500">
          {isVideo ? `${info.width}×${info.height} · ${info.fps ?? '?'} fps · ` : ''}
          {formatBytes(m.size)} · {timeAgo(m.created_at)}
        </div>
        {isVideo && (
          <Button size="sm" className="w-full" icon={Scissors} onClick={onExtract} disabled={info.readable === false}>
            Extract frames
          </Button>
        )}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ upload

function Dropzone({ project, kind, onUploaded }) {
  const [drag, setDrag] = useState(false)
  const [progress, setProgress] = useState(null)
  const inputRef = useRef(null)

  const upload = async (fileList) => {
    const exts = ACCEPT[kind].split(',')
    const files = [...fileList].filter((f) => exts.some((e) => f.name.toLowerCase().endsWith(e)))
    if (!files.length) {
      toast.error(`No supported ${kind === 'video' ? 'video' : 'image'} files selected`)
      return
    }
    setProgress(0)
    try {
      const res = await uploadFiles(project.name, files, setProgress)
      res.errors?.forEach((e) => toast.error(e))
      if (res.saved?.length) toast.success(`Uploaded ${res.saved.length} file${res.saved.length > 1 ? 's' : ''}`)
      onUploaded(res.saved || [])
    } catch (e) {
      toast.error(e.message)
    } finally {
      setProgress(null)
    }
  }

  const Icon = kind === 'video' ? Video : ImagePlus
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)
        upload(e.dataTransfer.files)
      }}
      onClick={() => progress === null && inputRef.current?.click()}
      className={cx(
        'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors',
        drag ? 'border-emerald-500 bg-emerald-500/5' : 'border-slate-800 bg-slate-900/50 hover:border-slate-700',
      )}
    >
      <input ref={inputRef} type="file" multiple={kind === 'photos'} accept={ACCEPT[kind]} hidden
             onChange={(e) => {
               upload(e.target.files)
               e.target.value = ''
             }} />
      {progress !== null ? (
        <div className="w-full max-w-sm space-y-3">
          <UploadCloud className="mx-auto size-8 text-emerald-400" />
          <Progress value={progress} />
          <p className="text-sm text-slate-400">Uploading… {Math.round(progress * 100)}%</p>
        </div>
      ) : (
        <>
          <div className="mb-3 rounded-full bg-slate-900 p-3 ring-1 ring-slate-800">
            <Icon className="size-6 text-slate-400" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            {kind === 'video' ? 'Drop a video file here or click to browse' : 'Drop photos here or click to browse'}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {kind === 'video' ? 'MP4, AVI, MKV, MOV, WebM — frames are extracted in the next step' : 'JPG, PNG, WebP, BMP — batch upload supported'}
          </p>
        </>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ RTSP

function RtspPanel({ project, onRecorded, onSnapshot }) {
  const [url, setUrl] = useState('')
  const [name, setName] = useState('camera1')
  const [stream, setStream] = useState(null)
  const [busy, setBusy] = useState(false)
  const [imgKey, setImgKey] = useState(0)

  useEffect(() => {
    api.streams(project.name).then((list) => list[0] && setStream(list[0])).catch(() => {})
  }, [project.name])

  useEffect(() => {
    if (!stream?.id) return
    const t = setInterval(async () => {
      try {
        setStream(await api.streamStatus(stream.id))
      } catch {
        setStream(null)
      }
    }, 1000)
    return () => clearInterval(t)
  }, [stream?.id])

  const connect = async () => {
    setBusy(true)
    try {
      const s = await api.openStream(project.name, url, name)
      setStream(s)
      setImgKey((k) => k + 1)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  const disconnect = async () => {
    if (stream.recording && !confirmAction('A recording is in progress. Stop it and disconnect?')) return
    try {
      await api.closeStream(stream.id)
    } catch {
      /* already gone */
    }
    setStream(null)
  }

  const toggleRecord = async () => {
    setBusy(true)
    try {
      if (stream.recording) {
        const entry = await api.recordStop(stream.id)
        toast.success(`Saved ${entry.filename}`)
        onRecorded(entry)
      } else {
        setStream(await api.recordStart(stream.id))
        toast('Recording started')
      }
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  const snap = async () => {
    try {
      await api.snapshot(stream.id)
      toast.success('Snapshot saved to extracted frames')
      onSnapshot()
    } catch (e) {
      toast.error(e.message)
    }
  }

  const live = stream?.status === 'live'
  return (
    <Card title="Live RTSP stream" icon={Radio}
          actions={stream && (
            <Badge tone={live ? 'emerald' : 'amber'} dot pulse={!live}>
              {stream.status}
            </Badge>
          )}>
      {!stream ? (
        <div className="grid gap-4 md:grid-cols-[1fr_200px_auto] md:items-end">
          <Field label="Stream URL" hint="rtsp://user:pass@host:554/stream — or a webcam index like 0">
            <Input placeholder="rtsp://192.168.1.10:554/stream1" value={url} onChange={(e) => setUrl(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && url && connect()} />
          </Field>
          <Field label="Stream name" hint="Used in recording file names">
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          <Button variant="primary" icon={Plug} loading={busy} disabled={!url.trim()} onClick={connect} className="mb-5">
            Connect
          </Button>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1fr_260px]">
          <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-black">
            <img key={imgKey} src={api.mjpegUrl(stream.id)} alt="Live stream" className="aspect-video w-full object-contain" />
            {!live && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/70 text-sm text-slate-300">
                <Loader2 className="size-6 animate-spin text-amber-400" />
                {stream.error || 'Connecting…'}
              </div>
            )}
            {stream.recording && (
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded-md bg-slate-950/80 px-2.5 py-1 text-xs font-semibold text-amber-400">
                <Circle className="rec-pulse size-2.5 fill-current" /> REC {formatDuration(stream.recording_seconds)}
              </div>
            )}
          </div>
          <div className="space-y-4">
            <dl className="space-y-1.5 rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs">
              {[
                ['Source', stream.url],
                ['Resolution', stream.width ? `${stream.width}×${stream.height}` : '—'],
                ['Measured FPS', stream.fps || '—'],
                ['Recording', stream.recording ? stream.recording_file : 'idle'],
                ['Frames written', stream.recording ? stream.frames_written : '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3">
                  <dt className="text-slate-500">{k}</dt>
                  <dd className="truncate text-right text-slate-300" title={String(v)}>{v}</dd>
                </div>
              ))}
            </dl>
            <Button size="lg" className="w-full" variant={stream.recording ? 'amber' : 'primary'} loading={busy}
                    icon={stream.recording ? Square : Circle} disabled={!live && !stream.recording} onClick={toggleRecord}>
              {stream.recording ? 'Stop recording' : 'Start recording'}
            </Button>
            <Button className="w-full" icon={Camera} disabled={!live} onClick={snap}>Snapshot frame</Button>
            <Button className="w-full" variant="dangerGhost" icon={Unplug} onClick={disconnect}>Disconnect</Button>
            <p className="flex gap-1.5 text-[11px] leading-relaxed text-slate-500">
              <Link2 className="mt-0.5 size-3 shrink-0" />
              Recordings save to raw_media/ as MP4. Recording continues if you switch tabs.
            </p>
          </div>
        </div>
      )}
    </Card>
  )
}

// ------------------------------------------------------------------ extraction

const FPS_PRESETS = [0.5, 1, 2, 5, 10]

function ExtractModal({ project, media, onClose, onDone }) {
  const [mode, setMode] = useState('fps')
  const [fps, setFps] = useState(1)
  const [interval, setIntervalN] = useState(30)
  const [format, setFormat] = useState('jpg')
  const [job, setJob] = useState(null)
  const esRef = useRef(null)

  useEffect(() => {
    if (!media) {
      esRef.current?.close()
      setJob(null)
    }
  }, [media])

  useEffect(() => () => esRef.current?.close(), [])

  if (!media) return null
  const info = media.info || {}
  const srcFps = info.fps || 25
  const frames = info.frame_count || 0
  const estimate = frames ? Math.floor(frames / (mode === 'fps' ? Math.max(1, srcFps / fps) : Math.max(1, interval))) + 1 : null
  const running = job && job.status === 'running'

  const start = async () => {
    try {
      const j = await api.extract(project.name, media.filename, { mode, fps: Number(fps), interval: Number(interval), format })
      setJob(j)
      const es = new EventSource(api.jobEventsUrl(j.id))
      esRef.current = es
      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data)
        setJob(data)
        if (data.status !== 'running') {
          es.close()
          if (data.status === 'completed') toast.success(`Extracted ${data.result?.extracted ?? data.current} frames`)
          else if (data.status === 'failed') toast.error(data.error || 'Extraction failed')
          onDone()
        }
      }
      es.onerror = () => es.close()
    } catch (e) {
      toast.error(e.message)
    }
  }

  return (
    <Modal
      open
      onClose={running ? () => {} : onClose}
      title="Extract frames"
      footer={
        running ? (
          <Button variant="dangerGhost" onClick={() => api.cancelJob(job.id)}>Cancel</Button>
        ) : job?.status === 'completed' ? (
          <Button variant="primary" onClick={onClose}>Done</Button>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button variant="primary" icon={Scissors} onClick={start}>Start extraction</Button>
          </>
        )
      }
    >
      <div className="space-y-5">
        <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
          <div className="truncate font-medium text-slate-200">{media.filename}</div>
          {info.width}×{info.height} · {info.fps ?? '?'} fps · {info.frame_count ?? '?'} frames · {formatDuration(info.duration)}
        </div>

        {job ? (
          <div className="space-y-3 py-2">
            <Progress value={job.progress} tone={job.status === 'failed' ? 'rose' : 'emerald'} />
            <div className="flex justify-between text-sm">
              <span className="text-slate-300">{job.status === 'failed' ? job.error : job.message}</span>
              <span className="tabular-nums text-slate-400">{job.current}{job.total ? ` / ~${job.total}` : ''}</span>
            </div>
          </div>
        ) : (
          <>
            <Field label="Extraction mode">
              <Segmented value={mode} onChange={setMode}
                         options={[{ value: 'fps', label: 'Extraction rate (FPS)' }, { value: 'interval', label: 'Every N-th frame' }]} />
            </Field>
            {mode === 'fps' ? (
              <Field label="Frames per second" hint={`0.5 FPS = one frame every 2 s. Source runs at ${srcFps} fps.`}>
                <div className="flex flex-wrap items-center gap-2">
                  {FPS_PRESETS.map((v) => (
                    <button key={v} onClick={() => setFps(v)}
                            className={cx('rounded-md border px-3 py-1.5 text-sm', Number(fps) === v
                              ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300' : 'border-slate-700 text-slate-300 hover:border-slate-600')}>
                      {v}
                    </button>
                  ))}
                  <Input type="number" min="0.01" step="0.1" value={fps} onChange={(e) => setFps(e.target.value)} className="w-24" />
                </div>
              </Field>
            ) : (
              <Field label="Interval (N)" hint="Keep frame 0, N, 2N, …">
                <Input type="number" min="1" value={interval} onChange={(e) => setIntervalN(e.target.value)} className="w-32" />
              </Field>
            )}
            <Field label="Image format">
              <Segmented value={format} onChange={setFormat} options={[{ value: 'jpg', label: 'JPG' }, { value: 'png', label: 'PNG (lossless)' }]} />
            </Field>
            {estimate != null && (
              <p className="text-xs text-slate-400">≈ <span className="font-semibold text-slate-200">{estimate}</span> frames will be saved to extracted_frames/</p>
            )}
          </>
        )}
      </div>
    </Modal>
  )
}
