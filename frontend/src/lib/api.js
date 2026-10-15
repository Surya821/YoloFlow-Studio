const BASE = import.meta.env.VITE_API_BASE || '/api'

async function request(method, path, body, opts = {}) {
  const init = { method, headers: {} }
  if (body instanceof FormData) {
    init.body = body
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }
  const res = await fetch(`${BASE}${path}`, { ...init, signal: opts.signal })
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!res.ok) {
    const detail = data?.detail
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail || `${res.status} ${res.statusText}`
    throw new Error(msg)
  }
  return data
}

const enc = encodeURIComponent
const p = (project) => `/projects/${enc(project)}`
const img = (imageId) => imageId.split('/').map(enc).join('/')

export const api = {
  get: (path, opts) => request('GET', path, undefined, opts),
  post: (path, body) => request('POST', path, body ?? {}),
  put: (path, body) => request('PUT', path, body),
  patch: (path, body) => request('PATCH', path, body),
  del: (path) => request('DELETE', path),

  catalog: () => request('GET', '/meta/catalog'),
  config: () => request('GET', '/config'),
  saveKaggle: (username, key) => request('PUT', '/config/kaggle', { username, key }),
  verifyKaggle: (body = {}) => request('POST', '/config/kaggle/verify', body),
  saveDefaults: (cfg) => request('PUT', '/config/defaults', cfg),

  projects: () => request('GET', '/projects'),
  createProject: (name) => request('POST', '/projects', { name }),
  deleteProject: (project) => request('DELETE', p(project)),
  stats: (project) => request('GET', `${p(project)}/stats`),

  addClass: (project, name, color) => request('POST', `${p(project)}/classes`, { name, color }),
  addClasses: (project, names) => request('POST', `${p(project)}/classes/bulk`, { names }),
  updateClass: (project, id, patch) => request('PATCH', `${p(project)}/classes/${id}`, patch),
  deleteClass: (project, id) => request('DELETE', `${p(project)}/classes/${id}`),

  media: (project) => request('GET', `${p(project)}/media`),
  deleteMedia: (project, filename) => request('DELETE', `${p(project)}/media/${enc(filename)}`),
  extract: (project, filename, body) => request('POST', `${p(project)}/media/${enc(filename)}/extract`, body),
  cancelJob: (id) => request('POST', `/jobs/${id}/cancel`),
  jobEventsUrl: (id) => `${BASE}/jobs/${id}/events`,

  fileUrl: (project, imageId) => `${BASE}${p(project)}/files/${img(imageId)}`,
  thumbUrl: (project, imageId) => `${BASE}${p(project)}/thumbs/${img(imageId)}`,

  images: (project) => request('GET', `${p(project)}/images`),
  deleteImage: (project, imageId) => request('DELETE', `${p(project)}/images/${img(imageId)}`),
  annotation: (project, imageId) => request('GET', `${p(project)}/annotations/${img(imageId)}`),
  saveAnnotation: (project, imageId, boxes) => request('PUT', `${p(project)}/annotations/${img(imageId)}`, { boxes }),
  clearAnnotation: (project, imageId) => request('DELETE', `${p(project)}/annotations/${img(imageId)}`),

  autolabelModels: (project) => request('GET', `${p(project)}/autolabel/models`),
  autolabelClasses: (project, model) => request('GET', `${p(project)}/autolabel/classes?model=${enc(model)}`),
  autolabel: (project, body) => request('POST', `${p(project)}/autolabel`, body),

  streams: (project) => request('GET', `${p(project)}/streams`),
  openStream: (project, url, name) => request('POST', `${p(project)}/streams`, { url, name }),
  streamStatus: (id) => request('GET', `/streams/${id}`),
  mjpegUrl: (id) => `${BASE}/streams/${id}/mjpeg`,
  recordStart: (id) => request('POST', `/streams/${id}/record/start`),
  recordStop: (id) => request('POST', `/streams/${id}/record/stop`),
  snapshot: (id) => request('POST', `/streams/${id}/snapshot`),
  closeStream: (id) => request('DELETE', `/streams/${id}`),

  models: (project) => request('GET', `/models${project ? `?project=${enc(project)}` : ''}`),
  deleteModel: (filename) => request('DELETE', `/models/custom/${enc(filename)}`),
  inferImage: (project, form) => request('POST', `${p(project)}/inference/image`, form),
  inferVideo: (project, form) => request('POST', `${p(project)}/inference/video`, form),
  videoResults: (project) => request('GET', `${p(project)}/inference/videos`),
  videoResultUrl: (project, id) => `${BASE}${p(project)}/inference/videos/${enc(id)}/file`,
  deleteVideoResult: (project, id) => request('DELETE', `${p(project)}/inference/videos/${enc(id)}`),
  liveSessions: (project) => request('GET', `${p(project)}/inference/live`),
  liveStart: (project, body) => request('POST', `${p(project)}/inference/live`, body),
  liveStatus: (id) => request('GET', `/inference/live/${id}`),
  liveUpdate: (id, patch) => request('PATCH', `/inference/live/${id}`, patch),
  liveMjpegUrl: (id) => `${BASE}/inference/live/${id}/mjpeg`,
  liveStop: (id) => request('DELETE', `/inference/live/${id}`),

  runs: (project) => request('GET', `${p(project)}/runs`),
  createRun: (project, backend, config) => request('POST', `${p(project)}/runs`, { backend, config }),
  run: (project, id) => request('GET', `${p(project)}/runs/${id}`),
  runLog: (project, id) => request('GET', `${p(project)}/runs/${id}/log`),
  runResults: (project, id) => request('GET', `${p(project)}/runs/${id}/results`),
  artifactUrl: (project, id, name) => `${BASE}${p(project)}/runs/${id}/artifacts/${enc(name)}`,
  cancelRun: (project, id) => request('POST', `${p(project)}/runs/${id}/cancel`),
  refreshRun: (project, id) => request('POST', `${p(project)}/runs/${id}/refresh`),
  deleteRun: (project, id) => request('DELETE', `${p(project)}/runs/${id}`),
}

/** Upload files with progress via XHR (fetch has no upload progress). */
export function uploadFiles(project, files, onProgress) {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  return xhrPost(`${BASE}${p(project)}/media`, form, onProgress)
}

/** Upload a YOLO model file (.pt / .onnx); the backend validates it before accepting. */
export function uploadModel(file, name, onProgress) {
  const form = new FormData()
  form.append('file', file)
  form.append('name', name || '')
  return xhrPost(`${BASE}/models/upload`, form, onProgress)
}

function xhrPost(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.onload = () => {
      let data = null
      try {
        data = JSON.parse(xhr.responseText)
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data)
      else reject(new Error(data?.detail || `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.send(form)
  })
}

export function formatBytes(n) {
  if (!n && n !== 0) return '—'
  const u = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024
    i++
  }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`
}

export function formatDuration(sec) {
  if (sec == null) return '—'
  sec = Math.round(sec)
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  return h ? `${h}h ${m}m` : m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`
}

export function timeAgo(ts) {
  if (!ts) return ''
  const d = Date.now() / 1000 - ts
  if (d < 60) return 'just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return new Date(ts * 1000).toLocaleDateString()
}
