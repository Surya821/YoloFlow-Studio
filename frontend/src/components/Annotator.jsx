import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import {
  Check, ChevronLeft, ChevronRight, Eraser, Hand, ImageOff, Keyboard, ListChecks, Loader2, Maximize, MousePointer2,
  Plus, Save, Search, Sparkles, Trash2, Undo2, Wand2, X, ZoomIn, ZoomOut,
} from 'lucide-react'
import { api } from '../lib/api'
import { Badge, Button, Card, EmptyState, IconButton, Input, Modal, Segmented, Select, Toggle, cx, confirmAction } from '../lib/ui'

const HANDLE_PX = 7
const HIT_PX = 8
const MIN_BOX_PX = 4
let _uid = 0
const uid = () => `b${++_uid}`

const hexToRgba = (hex, a) => {
  const h = (hex || '#94a3b8').replace('#', '')
  const n = parseInt(h.length === 3 ? h.replace(/(.)/g, '$1$1') : h, 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

const loadPref = (key, fallback) => {
  try {
    const v = localStorage.getItem(key)
    return v ? JSON.parse(v) : fallback
  } catch {
    return fallback
  }
}
const savePref = (key, v) => {
  try {
    localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

export default function Annotator({ project, catalog, onChanged }) {
  const pname = project.name
  const prefKey = `yf.ai.${pname}`

  // ------------------------------------------------ data state
  const [images, setImages] = useState([])
  const [filter, setFilter] = useState('all')
  const [currentId, setCurrentId] = useState(null)
  const [classes, setClasses] = useState(project.classes || [])
  const [activeClass, setActiveClass] = useState(0)
  const [boxes, setBoxes] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loadingImg, setLoadingImg] = useState(false)
  const [imgSize, setImgSize] = useState(null)
  const [view, setView] = useState({ scale: 1, ox: 0, oy: 0 })
  const [tool, setTool] = useState('draw')

  // ------------------------------------------------ AI state
  const [ai, setAi] = useState(() => loadPref(prefKey, { enabled: false, model: 'base:yolo11n.pt', conf: 0.3, classes: [] }))
  const [aiModels, setAiModels] = useState([])
  const [aiBusy, setAiBusy] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(null) // 'coco' | 'filter' | null
  const [filterOptions, setFilterOptions] = useState([])
  const [showKeys, setShowKeys] = useState(false)

  // ------------------------------------------------ refs (for event handlers / drawing)
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const imgRef = useRef(null)
  const viewRef = useRef(view)
  const boxesRef = useRef(boxes)
  const selectedRef = useRef(selectedId)
  const classesRef = useRef(classes)
  const dragRef = useRef(null)
  const draftRef = useRef(null)
  const hoverRef = useRef(null)
  const spaceRef = useRef(false)
  const historyRef = useRef([])
  const rafRef = useRef(0)
  const loadToken = useRef(0)
  const listRef = useRef(null)
  viewRef.current = view
  boxesRef.current = boxes
  selectedRef.current = selectedId
  classesRef.current = classes

  useEffect(() => savePref(prefKey, ai), [ai, prefKey])

  // ------------------------------------------------ loading
  const loadImages = useCallback(async () => {
    const list = await api.images(pname)
    setImages(list)
    return list
  }, [pname])

  useEffect(() => {
    loadImages()
      .then((list) => {
        if (list.length) setCurrentId((cur) => cur ?? (list.find((i) => !i.labeled) || list[0]).id)
      })
      .catch((e) => toast.error(e.message))
    api.autolabelModels(pname).then(setAiModels).catch(() => {})
  }, [pname, loadImages])

  const filtered = useMemo(
    () => images.filter((i) => filter === 'all' || (filter === 'labeled' ? i.labeled : !i.labeled)),
    [images, filter],
  )
  const current = images.find((i) => i.id === currentId) || null
  const hasImages = images.length > 0
  const counts = useMemo(() => ({ labeled: images.filter((i) => i.labeled).length, total: images.length }), [images])

  // ------------------------------------------------ view helpers
  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => drawRef.current?.())
  }, [])

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    const img = imgRef.current
    if (!wrap || !img) return
    const { clientWidth: cw, clientHeight: ch } = wrap
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight) * 0.94
    setView({ scale, ox: (cw - img.naturalWidth * scale) / 2, oy: (ch - img.naturalHeight * scale) / 2 })
  }, [])

  const zoomAt = useCallback((factor, sx, sy) => {
    setView((v) => {
      const scale = Math.min(40, Math.max(0.05, v.scale * factor))
      const k = scale / v.scale
      return { scale, ox: sx - (sx - v.ox) * k, oy: sy - (sy - v.oy) * k }
    })
  }, [])

  const zoomCenter = (factor) => {
    const wrap = wrapRef.current
    if (wrap) zoomAt(factor, wrap.clientWidth / 2, wrap.clientHeight / 2)
  }

  // ------------------------------------------------ AI pre-labeling
  const runAutolabel = useCallback(async (imageId, size, token) => {
    setAiBusy(true)
    try {
      const res = await api.autolabel(pname, {
        image_id: imageId, model: ai.model, conf: ai.conf, classes: ai.classes?.length ? ai.classes : null,
      })
      if (token !== undefined && token !== loadToken.current) return
      setClasses(res.classes)
      const W = size.w
      const H = size.h
      const suggestions = res.boxes.map((b) => ({
        id: uid(), cls: b.cls, suggested: true, conf: b.conf,
        x1: (b.cx - b.w / 2) * W, y1: (b.cy - b.h / 2) * H, x2: (b.cx + b.w / 2) * W, y2: (b.cy + b.h / 2) * H,
      }))
      setBoxes((prev) => [...prev.filter((b) => !b.suggested), ...suggestions])
      if (!suggestions.length) toast('AI found no matching objects', { duration: 1500 })
      if (res.classes.length !== classesRef.current.length) onChanged?.()
    } catch (e) {
      toast.error(`Auto-annotation failed: ${e.message}`)
    } finally {
      setAiBusy(false)
    }
  }, [pname, ai.model, ai.conf, ai.classes, onChanged])

  // Load image + annotation whenever the current image changes.
  useEffect(() => {
    if (!currentId) return
    const token = ++loadToken.current
    setLoadingImg(true)
    setSelectedId(null)
    historyRef.current = []
    const img = new Image()
    const imgPromise = new Promise((resolve, reject) => {
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('Could not load image'))
      img.src = api.fileUrl(pname, currentId)
    })
    Promise.all([imgPromise, api.annotation(pname, currentId)])
      .then(([loaded, ann]) => {
        if (token !== loadToken.current) return
        imgRef.current = loaded
        const size = { w: loaded.naturalWidth, h: loaded.naturalHeight }
        setImgSize(size)
        setBoxes(ann.boxes.map((b) => ({
          id: uid(), cls: b.cls,
          x1: (b.cx - b.w / 2) * size.w, y1: (b.cy - b.h / 2) * size.h,
          x2: (b.cx + b.w / 2) * size.w, y2: (b.cy + b.h / 2) * size.h,
        })))
        setDirty(false)
        fit()
        if (!ann.labeled && ai.enabled) runAutolabel(currentId, size, token)
      })
      .catch((e) => {
        if (token === loadToken.current) {
          imgRef.current = null
          setImgSize(null)
          toast.error(e.message)
        }
      })
      .finally(() => token === loadToken.current && setLoadingImg(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentId, pname])

  // Keep the active item visible in the list.
  useEffect(() => {
    listRef.current?.querySelector(`[data-id="${CSS.escape(currentId || '')}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [currentId, filter])

  // ------------------------------------------------ save & navigate
  const toNormalized = (list, size) =>
    list.map((b) => {
      const x1 = Math.min(b.x1, b.x2)
      const x2 = Math.max(b.x1, b.x2)
      const y1 = Math.min(b.y1, b.y2)
      const y2 = Math.max(b.y1, b.y2)
      return { cls: b.cls, cx: (x1 + x2) / 2 / size.w, cy: (y1 + y2) / 2 / size.h, w: (x2 - x1) / size.w, h: (y2 - y1) / size.h }
    })

  const save = useCallback(async ({ quiet } = {}) => {
    if (!currentId || !imgSize) return false
    setSaving(true)
    try {
      const res = await api.saveAnnotation(pname, currentId, toNormalized(boxesRef.current, imgSize))
      setImages((list) => list.map((i) => (i.id === currentId ? { ...i, labeled: true, box_count: res.box_count, split: res.split } : i)))
      setBoxes((list) => list.map((b) => (b.suggested ? { ...b, suggested: false } : b)))
      setDirty(false)
      if (!quiet) toast.success(`Saved · ${res.box_count} boxes → ${res.split}`, { duration: 1200 })
      onChanged?.()
      return true
    } catch (e) {
      toast.error(`Save failed: ${e.message}`)
      return false
    } finally {
      setSaving(false)
    }
  }, [currentId, imgSize, pname, onChanged])

  const navigate = useCallback(async (id) => {
    if (!id || id === currentId) return
    if (dirty && !(await save({ quiet: true }))) return
    setCurrentId(id)
  }, [currentId, dirty, save])

  const step = useCallback((delta) => {
    const list = filtered.length ? filtered : images
    const idx = list.findIndex((i) => i.id === currentId)
    const next = list[Math.max(0, Math.min(list.length - 1, (idx < 0 ? 0 : idx) + delta))]
    if (next) navigate(next.id)
  }, [filtered, images, currentId, navigate])

  // ------------------------------------------------ box editing
  const pushHistory = () => {
    historyRef.current.push(boxesRef.current.map((b) => ({ ...b })))
    if (historyRef.current.length > 60) historyRef.current.shift()
  }

  const undo = () => {
    const prev = historyRef.current.pop()
    if (prev) {
      setBoxes(prev)
      setDirty(true)
      setSelectedId(null)
    }
  }

  const deleteBox = (id) => {
    pushHistory()
    setBoxes((list) => list.filter((b) => b.id !== id))
    setSelectedId(null)
    setDirty(true)
  }

  const setBoxClass = (id, cls) => {
    pushHistory()
    setBoxes((list) => list.map((b) => (b.id === id ? { ...b, cls, suggested: false } : b)))
    setDirty(true)
  }

  const acceptSuggestions = () => {
    setBoxes((list) => list.map((b) => ({ ...b, suggested: false })))
    setDirty(true)
  }

  const discardSuggestions = () => setBoxes((list) => list.filter((b) => !b.suggested))

  // ------------------------------------------------ drawing
  const drawRef = useRef(null)
  drawRef.current = () => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const dpr = window.devicePixelRatio || 1
    const w = wrap.clientWidth
    const h = wrap.clientHeight
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
    }
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, w, h)
    const img = imgRef.current
    if (!img) return
    const { scale, ox, oy } = viewRef.current

    ctx.save()
    ctx.imageSmoothingEnabled = scale < 3
    ctx.drawImage(img, ox, oy, img.naturalWidth * scale, img.naturalHeight * scale)
    ctx.restore()

    const cls = classesRef.current
    const sel = selectedRef.current
    const toS = (x, y) => [ox + x * scale, oy + y * scale]
    ctx.font = '600 11px Inter, system-ui, sans-serif'
    ctx.textBaseline = 'middle'

    const ordered = [...boxesRef.current].sort((a, b) => (a.id === sel) - (b.id === sel))
    for (const b of ordered) {
      const color = cls[b.cls]?.color || '#94a3b8'
      const [x1, y1] = toS(Math.min(b.x1, b.x2), Math.min(b.y1, b.y2))
      const [x2, y2] = toS(Math.max(b.x1, b.x2), Math.max(b.y1, b.y2))
      const isSel = b.id === sel
      ctx.fillStyle = hexToRgba(color, isSel ? 0.22 : b.suggested ? 0.06 : 0.12)
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
      ctx.setLineDash(b.suggested ? [6, 4] : [])
      ctx.lineWidth = isSel ? 2.5 : 1.5
      ctx.strokeStyle = color
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.setLineDash([])

      const label = `${cls[b.cls]?.name ?? '?'}${b.suggested && b.conf ? ` ${Math.round(b.conf * 100)}%` : ''}`
      const tw = ctx.measureText(label).width + 10
      const ty = y1 - 18 < 0 ? y1 : y1 - 18
      ctx.fillStyle = color
      ctx.fillRect(x1 - (isSel ? 1.25 : 0.75), ty, tw, 18)
      ctx.fillStyle = '#020617'
      ctx.fillText(label, x1 + 5 - (isSel ? 1.25 : 0.75), ty + 9.5)

      if (isSel) {
        for (const [hx, hy] of handlePoints(x1, y1, x2, y2)) {
          ctx.fillStyle = '#f8fafc'
          ctx.strokeStyle = color
          ctx.lineWidth = 1.5
          ctx.fillRect(hx - HANDLE_PX / 2, hy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX)
          ctx.strokeRect(hx - HANDLE_PX / 2, hy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX)
        }
      }
    }

    const d = draftRef.current
    if (d) {
      const color = cls[activeClassRef.current]?.color || '#10b981'
      const [x1, y1] = toS(Math.min(d.x1, d.x2), Math.min(d.y1, d.y2))
      const [x2, y2] = toS(Math.max(d.x1, d.x2), Math.max(d.y1, d.y2))
      ctx.fillStyle = hexToRgba(color, 0.15)
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
      ctx.setLineDash([5, 3])
      ctx.lineWidth = 1.5
      ctx.strokeStyle = color
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.setLineDash([])
    }

    // Crosshair guide while in draw mode.
    const hv = hoverRef.current
    if (hv && tool === 'draw' && !spaceRef.current && (!dragRef.current || dragRef.current.type === 'draw')) {
      ctx.strokeStyle = 'rgba(148,163,184,0.45)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, hv.sy + 0.5)
      ctx.lineTo(w, hv.sy + 0.5)
      ctx.moveTo(hv.sx + 0.5, 0)
      ctx.lineTo(hv.sx + 0.5, h)
      ctx.stroke()
    }
  }

  const activeClassRef = useRef(activeClass)
  activeClassRef.current = activeClass

  useEffect(() => scheduleDraw(), [boxes, selectedId, view, classes, imgSize, tool, activeClass, scheduleDraw])

  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const ro = new ResizeObserver(() => scheduleDraw())
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [scheduleDraw, hasImages])

  // Wheel zoom needs a non-passive listener to preventDefault.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (e) => {
      e.preventDefault()
      const r = canvas.getBoundingClientRect()
      zoomAt(Math.pow(1.0015, -e.deltaY), e.clientX - r.left, e.clientY - r.top)
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [zoomAt, hasImages])

  // ------------------------------------------------ pointer interaction
  const localPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect()
    const sx = e.clientX - r.left
    const sy = e.clientY - r.top
    const v = viewRef.current
    return { sx, sy, x: (sx - v.ox) / v.scale, y: (sy - v.oy) / v.scale }
  }

  const clampToImg = (x, y) => {
    const img = imgRef.current
    return [Math.max(0, Math.min(img.naturalWidth, x)), Math.max(0, Math.min(img.naturalHeight, y))]
  }

  const screenRect = (b) => {
    const { scale, ox, oy } = viewRef.current
    return [ox + Math.min(b.x1, b.x2) * scale, oy + Math.min(b.y1, b.y2) * scale,
      ox + Math.max(b.x1, b.x2) * scale, oy + Math.max(b.y1, b.y2) * scale]
  }

  const hitHandle = (b, sx, sy) => {
    const [x1, y1, x2, y2] = screenRect(b)
    const names = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
    const pts = handlePoints(x1, y1, x2, y2)
    for (let i = 0; i < pts.length; i++) {
      if (Math.abs(sx - pts[i][0]) <= HIT_PX && Math.abs(sy - pts[i][1]) <= HIT_PX) return names[i]
    }
    // Edges (beyond handles) also resize.
    const inY = sy > y1 && sy < y2
    const inX = sx > x1 && sx < x2
    if (inY && Math.abs(sx - x1) <= 4) return 'w'
    if (inY && Math.abs(sx - x2) <= 4) return 'e'
    if (inX && Math.abs(sy - y1) <= 4) return 'n'
    if (inX && Math.abs(sy - y2) <= 4) return 's'
    return null
  }

  const hitBox = (sx, sy) => {
    let best = null
    let bestArea = Infinity
    for (const b of boxesRef.current) {
      const [x1, y1, x2, y2] = screenRect(b)
      if (sx >= x1 && sx <= x2 && sy >= y1 && sy <= y2) {
        const area = (x2 - x1) * (y2 - y1)
        if (area < bestArea) {
          best = b
          bestArea = area
        }
      }
    }
    return best
  }

  const CURSORS = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' }

  const onPointerDown = (e) => {
    if (!imgRef.current) return
    canvasRef.current.setPointerCapture(e.pointerId)
    const p = localPos(e)
    const v = viewRef.current
    if (e.button === 1 || e.button === 2 || spaceRef.current || tool === 'pan') {
      dragRef.current = { type: 'pan', sx: p.sx, sy: p.sy, ox: v.ox, oy: v.oy }
      return
    }
    if (e.button !== 0) return
    const sel = boxesRef.current.find((b) => b.id === selectedRef.current)
    const handle = sel && hitHandle(sel, p.sx, p.sy)
    if (handle) {
      dragRef.current = { type: 'resize', id: sel.id, handle, orig: { ...sel }, start: p, pushed: false }
      return
    }
    const hit = hitBox(p.sx, p.sy)
    if (hit && !(e.shiftKey || e.altKey)) {
      setSelectedId(hit.id)
      if (classesRef.current[hit.cls]) setActiveClass(hit.cls)
      dragRef.current = { type: 'move', id: hit.id, orig: { ...hit }, start: p, pushed: false }
      return
    }
    if (!classesRef.current.length) {
      toast.error('Add a class first (right panel)')
      return
    }
    setSelectedId(null)
    const [x, y] = clampToImg(p.x, p.y)
    dragRef.current = { type: 'draw', start: { x, y } }
    draftRef.current = { x1: x, y1: y, x2: x, y2: y }
  }

  const onPointerMove = (e) => {
    if (!imgRef.current) return
    const p = localPos(e)
    hoverRef.current = p
    const d = dragRef.current
    const canvas = canvasRef.current
    if (!d) {
      if (spaceRef.current || tool === 'pan') canvas.style.cursor = 'grab'
      else {
        const sel = boxesRef.current.find((b) => b.id === selectedRef.current)
        const handle = sel && hitHandle(sel, p.sx, p.sy)
        canvas.style.cursor = handle ? CURSORS[handle] : hitBox(p.sx, p.sy) ? 'move' : 'crosshair'
      }
      scheduleDraw()
      return
    }
    if (d.type === 'pan') {
      canvas.style.cursor = 'grabbing'
      setView((v) => ({ ...v, ox: d.ox + (p.sx - d.sx), oy: d.oy + (p.sy - d.sy) }))
      return
    }
    if (d.type === 'draw') {
      const [x, y] = clampToImg(p.x, p.y)
      draftRef.current = { x1: d.start.x, y1: d.start.y, x2: x, y2: y }
      scheduleDraw()
      return
    }
    if (!d.pushed) {
      pushHistory()
      d.pushed = true
    }
    const img = imgRef.current
    const W = img.naturalWidth
    const H = img.naturalHeight
    const dx = p.x - d.start.x
    const dy = p.y - d.start.y
    const o = d.orig
    let next
    if (d.type === 'move') {
      const bw = Math.abs(o.x2 - o.x1)
      const bh = Math.abs(o.y2 - o.y1)
      const nx = Math.max(0, Math.min(W - bw, Math.min(o.x1, o.x2) + dx))
      const ny = Math.max(0, Math.min(H - bh, Math.min(o.y1, o.y2) + dy))
      next = { x1: nx, y1: ny, x2: nx + bw, y2: ny + bh }
    } else {
      let x1 = Math.min(o.x1, o.x2)
      let x2 = Math.max(o.x1, o.x2)
      let y1 = Math.min(o.y1, o.y2)
      let y2 = Math.max(o.y1, o.y2)
      const [cxp, cyp] = clampToImg(p.x, p.y)
      if (d.handle.includes('w')) x1 = cxp
      if (d.handle.includes('e')) x2 = cxp
      if (d.handle.includes('n')) y1 = cyp
      if (d.handle.includes('s')) y2 = cyp
      next = { x1, y1, x2, y2 }
    }
    setBoxes((list) => list.map((b) => (b.id === d.id ? { ...b, ...next, suggested: false } : b)))
    setDirty(true)
  }

  const onPointerUp = (e) => {
    const d = dragRef.current
    dragRef.current = null
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) canvasRef.current.releasePointerCapture(e.pointerId)
    if (!d) return
    if (d.type === 'draw') {
      const r = draftRef.current
      draftRef.current = null
      const scale = viewRef.current.scale
      if (r && Math.abs(r.x2 - r.x1) * scale >= MIN_BOX_PX && Math.abs(r.y2 - r.y1) * scale >= MIN_BOX_PX) {
        pushHistory()
        const box = {
          id: uid(), cls: activeClassRef.current,
          x1: Math.min(r.x1, r.x2), y1: Math.min(r.y1, r.y2), x2: Math.max(r.x1, r.x2), y2: Math.max(r.y1, r.y2),
        }
        setBoxes((list) => [...list, box])
        setSelectedId(box.id)
        setDirty(true)
      }
      scheduleDraw()
    } else if (d.type === 'resize') {
      // Normalize flipped boxes after a resize drag.
      setBoxes((list) => list.map((b) => (b.id === d.id
        ? { ...b, x1: Math.min(b.x1, b.x2), x2: Math.max(b.x1, b.x2), y1: Math.min(b.y1, b.y2), y2: Math.max(b.y1, b.y2) }
        : b)))
    }
  }

  // ------------------------------------------------ keyboard
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || pickerOpen) return
      const k = e.key
      if (k === ' ') {
        spaceRef.current = e.type === 'keydown'
        if (canvasRef.current) canvasRef.current.style.cursor = spaceRef.current ? 'grab' : 'crosshair'
        e.preventDefault()
        return
      }
      if (e.type !== 'keydown') return
      if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') {
        e.preventDefault()
        save()
      } else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'z') {
        e.preventDefault()
        undo()
      } else if ((k === 'Delete' || k === 'Backspace') && selectedRef.current) {
        e.preventDefault()
        deleteBox(selectedRef.current)
      } else if (k === 'ArrowRight' || k.toLowerCase() === 'd') {
        step(1)
      } else if (k === 'ArrowLeft' || k.toLowerCase() === 'a') {
        step(-1)
      } else if (k === 'Enter') {
        e.preventDefault()
        acceptSuggestions()
        setTimeout(() => save({ quiet: true }).then((ok) => ok && step(1)), 0)
      } else if (k.toLowerCase() === 'f') {
        fit()
      } else if (k.toLowerCase() === 'h') {
        setTool((t) => (t === 'pan' ? 'draw' : 'pan'))
      } else if (k === 'Escape') {
        setSelectedId(null)
        dragRef.current = null
        draftRef.current = null
        scheduleDraw()
      } else if (/^[1-9]$/.test(k)) {
        const idx = Number(k) - 1
        if (idx < classesRef.current.length) {
          setActiveClass(idx)
          if (selectedRef.current) setBoxClass(selectedRef.current, idx)
        }
      } else if (k === '?') {
        setShowKeys((s) => !s)
      }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
    }
  })

  // Warn about unsaved changes when leaving the page.
  useEffect(() => {
    const handler = (e) => {
      if (dirty) {
        e.preventDefault()
        e.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])

  // ------------------------------------------------ class management
  const colorTimers = useRef({})
  const addClassNames = async (names) => {
    try {
      const list = await api.addClasses(pname, names)
      setClasses(list)
      onChanged?.()
      return list
    } catch (e) {
      toast.error(e.message)
    }
  }

  const renameClass = async (id, name) => {
    try {
      setClasses(await api.updateClass(pname, id, { name }))
      onChanged?.()
    } catch (e) {
      toast.error(e.message)
    }
  }

  const recolorClass = (id, color) => {
    setClasses((list) => list.map((c, i) => (i === id ? { ...c, color } : c)))
    clearTimeout(colorTimers.current[id])
    colorTimers.current[id] = setTimeout(() => api.updateClass(pname, id, { color }).catch((e) => toast.error(e.message)), 400)
  }

  const removeClass = async (id) => {
    const c = classes[id]
    if (!confirmAction(`Delete class "${c.name}"? All its boxes will be removed from every label file and higher class ids shift down.`)) return
    try {
      const list = await api.deleteClass(pname, id)
      setClasses(list)
      setBoxes((bs) => bs.filter((b) => b.cls !== id).map((b) => (b.cls > id ? { ...b, cls: b.cls - 1 } : b)))
      setActiveClass((a) => Math.max(0, a > id ? a - 1 : a === id ? 0 : a))
      loadImages()
      onChanged?.()
      toast.success(`Deleted "${c.name}"`)
    } catch (e) {
      toast.error(e.message)
    }
  }

  // ------------------------------------------------ image actions
  const clearLabel = async () => {
    if (!current?.labeled) {
      pushHistory()
      setBoxes([])
      return
    }
    if (!confirmAction('Remove this image from the dataset (delete its label file)?')) return
    await api.clearAnnotation(pname, currentId)
    setBoxes([])
    setDirty(false)
    setImages((list) => list.map((i) => (i.id === currentId ? { ...i, labeled: false, box_count: 0, split: null } : i)))
    onChanged?.()
  }

  const deleteImage = async () => {
    if (!confirmAction(`Permanently delete ${current.filename} and its label?`)) return
    const list = filtered.length ? filtered : images
    const idx = list.findIndex((i) => i.id === currentId)
    const next = list[idx + 1] || list[idx - 1]
    try {
      await api.deleteImage(pname, currentId)
      setImages((l) => l.filter((i) => i.id !== currentId))
      setDirty(false)
      setCurrentId(next?.id ?? null)
      if (!next) {
        imgRef.current = null
        setBoxes([])
      }
      onChanged?.()
    } catch (e) {
      toast.error(e.message)
    }
  }

  const openFilterPicker = async () => {
    try {
      setFilterOptions(await api.autolabelClasses(pname, ai.model))
      setPickerOpen('filter')
    } catch (e) {
      toast.error(e.message)
    }
  }

  // ------------------------------------------------ render
  const suggestions = boxes.filter((b) => b.suggested).length
  const selected = boxes.find((b) => b.id === selectedId)
  const idxInList = filtered.findIndex((i) => i.id === currentId)

  if (!images.length) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <EmptyState icon={ImageOff} title="No images to annotate">
          Upload photos or extract frames from a video in Media & Capture first.
        </EmptyState>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      {/* ------------------------------------------------ image list */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-slate-800 bg-slate-950">
        <div className="space-y-2 border-b border-slate-800 p-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{counts.labeled} / {counts.total} labeled</span>
            <span className="tabular-nums">{counts.total ? Math.round((counts.labeled / counts.total) * 100) : 0}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full bg-emerald-500" style={{ width: `${counts.total ? (counts.labeled / counts.total) * 100 : 0}%` }} />
          </div>
          <Segmented size="sm" value={filter} onChange={setFilter}
                     options={[{ value: 'all', label: 'All' }, { value: 'unlabeled', label: 'Todo' }, { value: 'labeled', label: 'Done' }]} />
        </div>
        <div ref={listRef} className="flex-1 overflow-y-auto p-1.5">
          {filtered.map((im) => (
            <button
              key={im.id}
              data-id={im.id}
              onClick={() => navigate(im.id)}
              className={cx(
                'flex w-full items-center gap-2.5 rounded-lg p-1.5 text-left transition-colors',
                im.id === currentId ? 'bg-slate-800 ring-1 ring-slate-700' : 'hover:bg-slate-900',
              )}
            >
              <img src={api.thumbUrl(pname, im.id)} loading="lazy" alt="" className="h-9 w-14 shrink-0 rounded bg-slate-900 object-cover" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-xs text-slate-200">{im.filename}</span>
                <span className="flex items-center gap-1.5 text-[10px] text-slate-500">
                  {im.labeled ? (
                    <>
                      <Check className="size-3 text-emerald-400" /> {im.box_count} boxes · {im.split}
                    </>
                  ) : (
                    'unlabeled'
                  )}
                </span>
              </span>
            </button>
          ))}
          {!filtered.length && <p className="p-4 text-center text-xs text-slate-500">No images in this filter.</p>}
        </div>
      </aside>

      {/* ------------------------------------------------ canvas */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 items-center gap-1 border-b border-slate-800 bg-slate-950 px-2">
          <IconButton icon={ChevronLeft} title="Previous (A / ←)" onClick={() => step(-1)} />
          <IconButton icon={ChevronRight} title="Next (D / →)" onClick={() => step(1)} />
          <div className="mx-2 min-w-0">
            <div className="truncate text-xs font-medium text-slate-200">{current?.filename}</div>
            <div className="text-[10px] text-slate-500">
              {idxInList + 1} / {filtered.length}{imgSize && ` · ${imgSize.w}×${imgSize.h}`}
              {current?.labeled && ` · ${current.split}`}
            </div>
          </div>
          <div className="mx-1 h-6 w-px bg-slate-800" />
          <IconButton icon={MousePointer2} title="Draw / select (H toggles)" active={tool === 'draw'} onClick={() => setTool('draw')} />
          <IconButton icon={Hand} title="Pan (hold Space)" active={tool === 'pan'} onClick={() => setTool('pan')} />
          <IconButton icon={ZoomOut} title="Zoom out" onClick={() => zoomCenter(1 / 1.25)} />
          <span className="w-12 text-center text-[11px] tabular-nums text-slate-400">{Math.round(view.scale * 100)}%</span>
          <IconButton icon={ZoomIn} title="Zoom in" onClick={() => zoomCenter(1.25)} />
          <IconButton icon={Maximize} title="Fit to screen (F)" onClick={fit} />
          <IconButton icon={Undo2} title="Undo (Ctrl+Z)" onClick={undo} />
          <div className="flex-1" />
          {aiBusy && <Badge tone="amber" dot pulse>AI detecting…</Badge>}
          <IconButton icon={Keyboard} title="Shortcuts (?)" onClick={() => setShowKeys(true)} />
          <IconButton icon={Eraser} title={current?.labeled ? 'Remove from dataset' : 'Clear boxes'} onClick={clearLabel} />
          <IconButton icon={Trash2} title="Delete image" className="hover:!bg-rose-600/20 hover:!text-rose-400" onClick={deleteImage} />
          <Button size="sm" variant={dirty ? 'primary' : 'secondary'} icon={Save} loading={saving} onClick={() => save()} className="ml-1">
            {dirty ? 'Save' : current?.labeled ? 'Saved' : 'Save'}
          </Button>
        </div>

        <div ref={wrapRef} className="relative min-h-0 flex-1 overflow-hidden">
          <canvas
            ref={canvasRef}
            className="absolute inset-0 touch-none select-none"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              hoverRef.current = null
              scheduleDraw()
            }}
            onContextMenu={(e) => e.preventDefault()}
          />
          {loadingImg && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <Loader2 className="size-6 animate-spin text-slate-500" />
            </div>
          )}
          {suggestions > 0 && (
            <div className="absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-3 rounded-lg border border-amber-500/30 bg-slate-900/95 px-3 py-2 text-xs shadow-lg">
              <Sparkles className="size-4 text-amber-400" />
              <span className="text-slate-200">{suggestions} AI suggestion{suggestions > 1 ? 's' : ''} (dashed)</span>
              <Button size="sm" variant="primary" icon={Check} onClick={acceptSuggestions}>Keep</Button>
              <Button size="sm" variant="ghost" icon={X} onClick={discardSuggestions}>Discard</Button>
            </div>
          )}
          {!classes.length && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-300">
              Add a class in the right panel to start drawing boxes.
            </div>
          )}
        </div>
      </section>

      {/* ------------------------------------------------ right panel */}
      <aside className="w-80 shrink-0 space-y-4 overflow-y-auto border-l border-slate-800 bg-slate-950 p-3">
        <ClassPanel
          classes={classes}
          active={activeClass}
          onActivate={(i) => {
            setActiveClass(i)
            if (selectedId) setBoxClass(selectedId, i)
          }}
          onAdd={(name, color) => api.addClass(pname, name, color).then((l) => {
            setClasses(l)
            setActiveClass(l.length - 1)
            onChanged?.()
          }).catch((e) => toast.error(e.message))}
          onRename={renameClass}
          onRecolor={recolorClass}
          onDelete={removeClass}
          onOpenCoco={() => setPickerOpen('coco')}
        />

        <Card title="AI auto-annotation" icon={Wand2} bodyClass="space-y-3">
          <Toggle checked={ai.enabled} onChange={(enabled) => setAi({ ...ai, enabled })} label="Enable AI pre-labeling"
                  description="Runs on unannotated images when you open them" />
          <div>
            <span className="mb-1 block text-xs text-slate-400">Base model</span>
            <Select value={ai.model} onChange={(e) => setAi({ ...ai, model: e.target.value, classes: [] })}>
              {(aiModels.length ? aiModels : [{ key: ai.model, label: ai.model }]).map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </Select>
            <span className="mt-1 block text-[10px] text-slate-500">COCO weights download on first use. Trained project runs appear here too.</span>
          </div>
          <div>
            <div className="mb-1 flex justify-between text-xs text-slate-400">
              <span>Confidence threshold</span>
              <span className="tabular-nums text-slate-200">{ai.conf.toFixed(2)}</span>
            </div>
            <input type="range" min="0.05" max="0.95" step="0.05" value={ai.conf} className="w-full"
                   onChange={(e) => setAi({ ...ai, conf: Number(e.target.value) })} />
          </div>
          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
              <span>Detect classes</span>
              <button className="text-emerald-400 hover:underline" onClick={openFilterPicker}>Edit</button>
            </div>
            <div className="flex flex-wrap gap-1">
              {ai.classes?.length ? (
                ai.classes.slice(0, 12).map((c) => <Badge key={c}>{c}</Badge>)
              ) : (
                <span className="text-[11px] text-slate-500">All model classes</span>
              )}
              {ai.classes?.length > 12 && <Badge>+{ai.classes.length - 12}</Badge>}
            </div>
          </div>
          <Button className="w-full" icon={Sparkles} loading={aiBusy} disabled={!imgSize}
                  onClick={() => runAutolabel(currentId, imgSize, loadToken.current)}>
            Run on this image
          </Button>
        </Card>

        <Card title={`Boxes (${boxes.length})`} icon={ListChecks} bodyClass="p-2">
          {!boxes.length ? (
            <p className="p-2 text-xs text-slate-500">Drag on the image to draw a box. Saving with no boxes marks the image as background.</p>
          ) : (
            <ul className="space-y-1">
              {boxes.map((b, i) => (
                <li key={b.id}
                    onClick={() => setSelectedId(b.id)}
                    className={cx('flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5', b.id === selectedId ? 'bg-slate-800' : 'hover:bg-slate-900')}>
                  <span className="size-2.5 shrink-0 rounded-sm" style={{ background: classes[b.cls]?.color }} />
                  <span className="w-5 text-[10px] tabular-nums text-slate-500">#{i + 1}</span>
                  <select value={b.cls} onClick={(e) => e.stopPropagation()} onChange={(e) => setBoxClass(b.id, Number(e.target.value))}
                          className="min-w-0 flex-1 rounded border-none bg-transparent text-xs text-slate-200 focus:outline-none">
                    {classes.map((c, ci) => <option key={ci} value={ci} className="bg-slate-900">{c.name}</option>)}
                  </select>
                  {b.suggested && <Badge tone="amber">{Math.round((b.conf || 0) * 100)}%</Badge>}
                  <button onClick={(e) => {
                    e.stopPropagation()
                    deleteBox(b.id)
                  }} className="rounded p-1 text-slate-500 hover:bg-rose-600/20 hover:text-rose-400" title="Delete box">
                    <Trash2 className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}
          {selected && (
            <p className="px-2 pt-2 text-[10px] tabular-nums text-slate-500">
              Selected: {Math.round(Math.abs(selected.x2 - selected.x1))}×{Math.round(Math.abs(selected.y2 - selected.y1))} px · Del to remove
            </p>
          )}
        </Card>
      </aside>

      <ClassPicker
        open={pickerOpen === 'coco'}
        title="Add standard COCO classes"
        options={catalog?.coco_classes || []}
        initial={[]}
        disabled={classes.map((c) => c.name.toLowerCase())}
        confirmLabel={(n) => `Add ${n} class${n === 1 ? '' : 'es'}`}
        onClose={() => setPickerOpen(null)}
        onConfirm={async (names) => {
          await addClassNames(names)
          setPickerOpen(null)
        }}
      />
      <ClassPicker
        open={pickerOpen === 'filter'}
        title="Classes to auto-detect"
        options={filterOptions}
        initial={ai.classes || []}
        hint="Leave empty to keep every class the model detects. Detected classes missing from the project are added automatically."
        quickSelect={classes.map((c) => c.name)}
        confirmLabel={(n) => (n ? `Use ${n} class${n === 1 ? '' : 'es'}` : 'Use all classes')}
        onClose={() => setPickerOpen(null)}
        onConfirm={(names) => {
          setAi({ ...ai, classes: names })
          setPickerOpen(null)
        }}
      />
      <ShortcutsModal open={showKeys} onClose={() => setShowKeys(false)} />
    </div>
  )
}

function handlePoints(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  return [[x1, y1], [mx, y1], [x2, y1], [x2, my], [x2, y2], [mx, y2], [x1, y2], [x1, my]]
}

// ------------------------------------------------------------------ class panel

function ClassPanel({ classes, active, onActivate, onAdd, onRename, onRecolor, onDelete, onOpenCoco }) {
  const [name, setName] = useState('')
  const [color, setColor] = useState('#10B981')
  const [editing, setEditing] = useState(null)
  const [editName, setEditName] = useState('')

  const submit = () => {
    if (!name.trim()) return
    onAdd(name.trim(), color)
    setName('')
  }

  return (
    <Card title={`Classes (${classes.length})`} icon={ListChecks} bodyClass="space-y-3"
          actions={<Button size="sm" variant="ghost" onClick={onOpenCoco}>+ COCO</Button>}>
      <ul className="space-y-1">
        {classes.map((c, i) => (
          <li key={`${i}-${c.name}`}
              className={cx('group flex items-center gap-2 rounded-md border px-2 py-1.5',
                i === active ? 'border-slate-600 bg-slate-800' : 'border-transparent hover:bg-slate-900')}>
            <input type="color" value={c.color} onChange={(e) => onRecolor(i, e.target.value)} className="size-5 shrink-0" title="Class color" />
            {editing === i ? (
              <input
                autoFocus
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onBlur={() => {
                  if (editName.trim() && editName !== c.name) onRename(i, editName)
                  setEditing(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur()
                  if (e.key === 'Escape') setEditing(null)
                }}
                className="min-w-0 flex-1 rounded bg-slate-950 px-1.5 py-0.5 text-sm text-slate-100 outline-none ring-1 ring-emerald-500"
              />
            ) : (
              <button className="min-w-0 flex-1 truncate text-left text-sm text-slate-200" onClick={() => onActivate(i)}
                      onDoubleClick={() => {
                        setEditing(i)
                        setEditName(c.name)
                      }} title="Click to select · double-click to rename">
                {c.name}
              </button>
            )}
            {i < 9 && <kbd className="rounded bg-slate-950 px-1.5 text-[10px] text-slate-500 ring-1 ring-slate-800">{i + 1}</kbd>}
            <button onClick={() => onDelete(i)} title="Delete class"
                    className="rounded p-0.5 text-slate-600 opacity-0 hover:text-rose-400 group-hover:opacity-100">
              <X className="size-3.5" />
            </button>
          </li>
        ))}
        {!classes.length && <li className="text-xs text-slate-500">No classes yet.</li>}
      </ul>
      <div className="flex gap-2">
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} className="h-9 w-9 shrink-0" title="New class color" />
        <Input placeholder="New class (e.g. Crack)" value={name} onChange={(e) => setName(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && submit()} />
        <Button icon={Plus} onClick={submit} disabled={!name.trim()} title="Add class" />
      </div>
    </Card>
  )
}

// ------------------------------------------------------------------ pickers

function ClassPicker({ open, title, options, initial, disabled = [], hint, quickSelect, confirmLabel, onClose, onConfirm }) {
  const [selected, setSelected] = useState(new Set())
  const [q, setQ] = useState('')
  useEffect(() => {
    if (open) {
      setSelected(new Set(initial))
      setQ('')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const shown = options.filter((o) => o.toLowerCase().includes(q.toLowerCase()))
  const toggle = (o) => setSelected((s) => {
    const n = new Set(s)
    n.has(o) ? n.delete(o) : n.add(o)
    return n
  })

  return (
    <Modal open={open} onClose={onClose} title={title} width="max-w-2xl"
           footer={
             <>
               <Button variant="ghost" onClick={() => setSelected(new Set())}>Clear</Button>
               <Button variant="primary" onClick={() => onConfirm([...selected])}
                       disabled={!quickSelect && selected.size === 0}>
                 {confirmLabel(selected.size)}
               </Button>
             </>
           }>
      <div className="space-y-3">
        {hint && <p className="text-xs text-slate-500">{hint}</p>}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-slate-500" />
            <Input autoFocus placeholder="Search classes…" value={q} onChange={(e) => setQ(e.target.value)} className="pl-8" />
          </div>
          {quickSelect && (
            <Button variant="secondary" onClick={() => setSelected(new Set(options.filter((o) => quickSelect.some((p) => p.toLowerCase() === o.toLowerCase()))))}>
              Match project classes
            </Button>
          )}
        </div>
        <div className="grid max-h-[50vh] grid-cols-2 gap-1 overflow-y-auto sm:grid-cols-3">
          {shown.map((o) => {
            const isDisabled = disabled.includes(o.toLowerCase())
            const checked = isDisabled || selected.has(o)
            return (
              <label key={o}
                     className={cx('flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm',
                       isDisabled ? 'cursor-default text-slate-500' : 'text-slate-200 hover:bg-slate-800')}>
                <input type="checkbox" checked={checked} disabled={isDisabled} onChange={() => toggle(o)}
                       className="size-3.5 accent-emerald-600" />
                {o}
              </label>
            )
          })}
        </div>
      </div>
    </Modal>
  )
}

function ShortcutsModal({ open, onClose }) {
  const rows = [
    ['Drag on image', 'Draw a box with the active class'],
    ['Drag box / handles', 'Move / resize'],
    ['1 – 9', 'Select class (re-labels the selected box)'],
    ['Delete / Backspace', 'Delete selected box'],
    ['A / ←,  D / →', 'Previous / next image (auto-saves)'],
    ['Enter', 'Keep AI suggestions, save and go next'],
    ['Ctrl + S', 'Save'],
    ['Ctrl + Z', 'Undo'],
    ['Wheel', 'Zoom at cursor'],
    ['Space + drag, right/middle drag', 'Pan'],
    ['F', 'Fit to screen'],
    ['H', 'Toggle pan tool'],
    ['Shift + drag over a box', 'Draw a nested box'],
    ['Esc', 'Deselect'],
  ]
  return (
    <Modal open={open} onClose={onClose} title="Keyboard shortcuts">
      <dl className="divide-y divide-slate-800 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-2">
            <dt><kbd className="rounded bg-slate-950 px-2 py-0.5 text-xs text-slate-300 ring-1 ring-slate-700">{k}</kbd></dt>
            <dd className="text-right text-slate-400">{v}</dd>
          </div>
        ))}
      </dl>
    </Modal>
  )
}
