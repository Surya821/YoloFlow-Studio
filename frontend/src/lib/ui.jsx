import { useEffect } from 'react'
import { Loader2, X } from 'lucide-react'

export const cx = (...c) => c.filter(Boolean).join(' ')

const VARIANTS = {
  primary: 'bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500/50',
  secondary: 'bg-slate-800 hover:bg-slate-700 text-slate-100 border-slate-700',
  ghost: 'bg-transparent hover:bg-slate-800 text-slate-300 border-transparent',
  danger: 'bg-rose-600 hover:bg-rose-500 text-white border-rose-500/50',
  dangerGhost: 'bg-transparent hover:bg-rose-600/15 text-rose-400 border-transparent',
  amber: 'bg-amber-500 hover:bg-amber-400 text-slate-950 border-amber-400/50',
}

export function Button({ variant = 'secondary', size = 'md', loading, icon: Icon, children, className, ...props }) {
  const sizes = { sm: 'h-7 px-2.5 text-xs gap-1.5', md: 'h-9 px-3.5 text-sm gap-2', lg: 'h-11 px-5 text-sm gap-2' }
  return (
    <button
      {...props}
      disabled={props.disabled || loading}
      className={cx(
        'inline-flex items-center justify-center rounded-lg border font-medium transition-colors',
        'disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-emerald-500',
        VARIANTS[variant],
        sizes[size],
        className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" /> : Icon ? <Icon className={size === 'sm' ? 'size-3.5' : 'size-4'} /> : null}
      {children}
    </button>
  )
}

export function IconButton({ icon: Icon, title, className, active, ...props }) {
  return (
    <button
      {...props}
      title={title}
      aria-label={title}
      className={cx(
        'inline-flex size-8 items-center justify-center rounded-md transition-colors disabled:opacity-40',
        active ? 'bg-emerald-600/20 text-emerald-400' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100',
        className,
      )}
    >
      <Icon className="size-4" />
    </button>
  )
}

export function Card({ title, icon: Icon, actions, children, className, bodyClass }) {
  return (
    <section className={cx('rounded-xl border border-slate-800 bg-slate-900', className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            {Icon && <Icon className="size-4 text-slate-400" />}
            {title}
          </h3>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx('p-4', bodyClass)}>{children}</div>
    </section>
  )
}

const TONES = {
  slate: 'bg-slate-800 text-slate-300 border-slate-700',
  emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  amber: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  rose: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
  sky: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
}

export function Badge({ tone = 'slate', children, className, dot, pulse }) {
  return (
    <span className={cx('inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 text-[11px] font-medium', TONES[tone], className)}>
      {dot && <span className={cx('size-1.5 rounded-full bg-current', pulse && 'rec-pulse')} />}
      {children}
    </span>
  )
}

export function Field({ label, hint, children, className }) {
  return (
    <label className={cx('block', className)}>
      <span className="mb-1.5 block text-xs font-medium text-slate-400">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-slate-500">{hint}</span>}
    </label>
  )
}

export const inputClass =
  'h-9 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 text-sm text-slate-100 placeholder:text-slate-600 ' +
  'focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 disabled:opacity-50'

export function Input(props) {
  return <input {...props} className={cx(inputClass, props.className)} />
}

export function Select({ children, ...props }) {
  return (
    <select {...props} className={cx(inputClass, 'pr-8', props.className)}>
      {children}
    </select>
  )
}

export function Segmented({ value, onChange, options, size = 'md' }) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-slate-800 bg-slate-950 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          title={o.title}
          className={cx(
            'rounded-md font-medium transition-colors',
            size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
            value === o.value ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({ checked, onChange, label, description }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="flex w-full items-start gap-3 text-left">
      <span
        className={cx(
          'relative mt-0.5 inline-flex h-5 w-9 shrink-0 rounded-full border transition-colors',
          checked ? 'border-emerald-500 bg-emerald-600' : 'border-slate-700 bg-slate-800',
        )}
      >
        <span className={cx('absolute top-0.5 size-3.5 rounded-full bg-white transition-all', checked ? 'left-[18px]' : 'left-0.5')} />
      </span>
      {(label || description) && (
        <span>
          {label && <span className="block text-sm font-medium text-slate-200">{label}</span>}
          {description && <span className="block text-xs text-slate-500">{description}</span>}
        </span>
      )}
    </button>
  )
}

export function Progress({ value, tone = 'emerald', className }) {
  const color = { emerald: 'bg-emerald-500', amber: 'bg-amber-500', rose: 'bg-rose-500' }[tone]
  return (
    <div className={cx('h-2 w-full overflow-hidden rounded-full bg-slate-800', className)}>
      <div className={cx('h-full rounded-full transition-all duration-300', color)} style={{ width: `${Math.max(0, Math.min(1, value || 0)) * 100}%` }} />
    </div>
  )
}

export function Modal({ open, onClose, title, children, footer, width = 'max-w-lg' }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose?.()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className={cx('flex max-h-[90vh] w-full flex-col rounded-xl border border-slate-800 bg-slate-900 shadow-2xl', width)}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-slate-800 px-5 py-3.5">
          <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
          <IconButton icon={X} title="Close" onClick={onClose} />
        </header>
        <div className="overflow-y-auto p-5">{children}</div>
        {footer && <footer className="flex justify-end gap-2 border-t border-slate-800 px-5 py-3">{footer}</footer>}
      </div>
    </div>
  )
}

export function EmptyState({ icon: Icon, title, children, action }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 px-6 py-12 text-center">
      {Icon && (
        <div className="mb-3 rounded-full bg-slate-900 p-3 ring-1 ring-slate-800">
          <Icon className="size-6 text-slate-500" />
        </div>
      )}
      <p className="text-sm font-medium text-slate-200">{title}</p>
      {children && <p className="mt-1 max-w-sm text-xs text-slate-500">{children}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function Stat({ label, value, sub, icon: Icon }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center justify-between text-xs font-medium text-slate-400">
        {label}
        {Icon && <Icon className="size-4 text-slate-500" />}
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums text-slate-50">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

/** Confirm via a lightweight native dialog (deletions are rose-colored at call sites). */
export const confirmAction = (msg) => window.confirm(msg)
