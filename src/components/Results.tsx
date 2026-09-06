import { useEffect, useState } from 'react'
import { AnimatePresence, animate, motion, useReducedMotion } from 'framer-motion'
import type { AnswerRecord, ToolResult } from '../lib/api'
import { IconClock, IconRuler, IconTarget, IconChange, IconX } from './Icons'
import { ease, panelIn, popIn, spring, stagger } from '../lib/motion'

/**
 * Headline figures and the maps behind them.
 *
 * Each card is read straight off the `AnswerRecord`, never recomputed here. A dashboard
 * that does its own arithmetic on the way to the screen is a second implementation of the
 * measurement, and the two drift.
 */

function fmtArea(m2: number): [string, string] {
  if (Math.abs(m2) >= 1_000_000) return [(m2 / 1_000_000).toFixed(2), 'km²']
  if (Math.abs(m2) >= 10_000) return [(m2 / 10_000).toFixed(2), 'ha']
  return [Math.round(m2).toLocaleString(), 'm²']
}

/**
 * Counts a figure up from zero on the way in.
 *
 * The animation decides only how the number *arrives*; it never decides what it is. The
 * settled state is the caller's own string, restored verbatim on completion, so a
 * rounding difference in the interpolation can never survive into what is read off the
 * screen. Anything that does not parse as a number is rendered untouched.
 */
function CountUp({ text, delay = 0 }: { text: string; delay?: number }) {
  const reduced = useReducedMotion()
  const [display, setDisplay] = useState(text)

  useEffect(() => {
    const match = /^([^\d]*)([\d,]+(?:\.\d+)?)(.*)$/.exec(text)
    const target = match ? Number(match[2].replace(/,/g, '')) : NaN
    if (reduced || !match || !Number.isFinite(target)) {
      setDisplay(text)
      return
    }
    const decimals = match[2].includes('.') ? match[2].split('.')[1].length : 0
    const controls = animate(0, target, {
      duration: 0.85,
      delay,
      ease,
      onUpdate: (value) =>
        setDisplay(
          `${match[1]}${value.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals,
          })}${match[3]}`,
        ),
      onComplete: () => setDisplay(text),
    })
    return () => controls.stop()
  }, [text, delay, reduced])

  return <>{display}</>
}

type Kpi = { label: string; value: string; unit?: string; tone?: string; icon: () => JSX.Element }

function kpisFor(record: AnswerRecord, result: ToolResult): Kpi[] {
  const cards: Kpi[] = []
  const deltas = Object.entries(record.area_deltas)

  if (deltas.length > 0) {
    const [name, delta] = deltas[0]
    // No transform means no ground area. Show the change in share of the scene rather
    // than a hectare figure the imagery cannot support.
    const signed = delta.absolute_m2 ?? ((delta.fraction_t2 ?? 0) - (delta.fraction_t1 ?? 0))
    const [value, unit] =
      delta.absolute_m2 != null
        ? fmtArea(Math.abs(delta.absolute_m2))
        : [(Math.abs(signed) * 100).toFixed(1), 'pp of scene']
    cards.push({
      label: `${name.replace(/_/g, ' ')} change`,
      value: `${signed >= 0 ? '+' : '−'}${value}`,
      unit,
      tone: delta.trend === 'increased' ? 'up' : delta.trend === 'decreased' ? 'down' : 'flat',
      icon: IconChange,
    })
    cards.push({
      label: 'relative change',
      value: `${(Math.abs(delta.relative) * 100).toFixed(1)}`,
      unit: '%',
      tone: delta.trend === 'increased' ? 'up' : delta.trend === 'decreased' ? 'down' : 'flat',
      icon: IconRuler,
    })
  } else {
    const proportions = Object.entries(record.class_proportions).sort((a, b) => b[1] - a[1])
    for (const [name, value] of proportions.slice(0, 2)) {
      cards.push({
        label: `${name.replace(/_/g, ' ')} cover`,
        value: (value * 100).toFixed(1),
        unit: '%',
        icon: IconRuler,
      })
    }
  }

  // With one signal there is nothing to agree with, so the card says so rather than
  // showing a dash that reads as a failed measurement.
  cards.push({
    label: result.confidence != null ? 'agreement' : 'agreement · none',
    value: result.confidence != null ? (result.confidence * 100).toFixed(0) : 'n/a',
    unit: result.confidence != null ? '%' : undefined,
    tone: result.confidence == null ? 'none' : result.confidence >= 0.75 ? 'up' : 'flat',
    icon: IconTarget,
  })
  cards.push({
    label: 'latency',
    value: result.latency_ms.toFixed(0),
    unit: 'ms',
    icon: IconClock,
  })

  return cards.slice(0, 4)
}

export function KpiRow({ record, result }: { record: AnswerRecord; result: ToolResult }) {
  const cards = kpisFor(record, result)
  return (
    <motion.div className="kpis" variants={stagger(0.08)} initial="hidden" animate="show">
      {cards.map((card, index) => {
        const Icon = card.icon
        return (
          <motion.div
            className={`kpi ${card.tone ?? ''}`}
            key={card.label}
            variants={panelIn}
            whileHover={{ y: -3, transition: spring }}
          >
            <div className="kpi-top">
              <span className="kpi-label">{card.label}</span>
              <Icon />
            </div>
            <div className="kpi-value">
              <CountUp text={card.value} delay={0.1 + index * 0.08} />
              {card.unit && <span className="kpi-unit">{card.unit}</span>}
            </div>
            <span className="kpi-sheen" />
          </motion.div>
        )
      })}
    </motion.div>
  )
}

/**
 * The maps the answer was measured from.
 *
 * Present only when the tool actually rendered them. An empty placeholder tile would
 * imply the system produced evidence it did not.
 *
 * A tile opens into a lightbox on the same element rather than a new one, so the map the
 * eye was already on is the map that grows -- there is no moment where the reader has to
 * find their tile again in a bigger picture.
 */
export function EvidenceGrid({ result }: { result: ToolResult }) {
  const [open, setOpen] = useState<string | null>(null)
  const fileOf = (path: string | null): string | null => path?.split('/').pop() ?? null

  const tiles: { key: string; title: string; file: string | null; note: string }[] = [
    {
      key: 'change',
      title: 'Change map',
      file: fileOf(result.evidence.mask_path),
      note: 'green gained · red lost · slate unchanged',
    },
    ...Object.entries(result.evidence.index_maps).map(([name, path]) => ({
      key: name,
      title: name.replace(/_t1$/, ' — date 1').replace(/_t2$/, ' — date 2').replace(/_/g, ' '),
      file: fileOf(String(path)),
      note: 'measured mask over the scene',
    })),
    {
      key: 'overlay',
      title: 'Class overlay',
      file: fileOf(result.evidence.overlay_path),
      note: 'every pixel counted, by class',
    },
  ].filter((tile) => tile.file)

  if (tiles.length === 0) return null
  const active = tiles.find((tile) => tile.key === open) ?? null

  return (
    <>
      <motion.div className="evidence-grid" variants={stagger(0.07)} initial="hidden" animate="show">
        {tiles.map((tile) => (
          <motion.figure
            className="etile"
            key={tile.key}
            layoutId={`evidence-${tile.key}`}
            variants={popIn}
            whileHover={{ y: -4, transition: spring }}
            onClick={() => setOpen(tile.key)}
          >
            <div className="etile-head">{tile.title}</div>
            <motion.img src={`/api/images/evidence/${tile.file}`} alt={tile.title} />
            <figcaption>{tile.note}</figcaption>
          </motion.figure>
        ))}
      </motion.div>

      <AnimatePresence>
        {active && (
          <motion.div
            className="lightbox"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(null)}
          >
            <motion.figure className="etile lb" layoutId={`evidence-${active.key}`}>
              <div className="etile-head">
                {active.title}
                <button className="ghost-btn" onClick={() => setOpen(null)}><IconX /></button>
              </div>
              <motion.img src={`/api/images/evidence/${active.file}`} alt={active.title} />
              <figcaption>{active.note}</figcaption>
            </motion.figure>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
