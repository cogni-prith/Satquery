import type { AnswerRecord, ToolResult } from '../lib/api'
import { IconClock, IconRuler, IconTarget, IconChange } from './Icons'

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
    <div className="kpis">
      {cards.map((card, index) => {
        const Icon = card.icon
        return (
          <div className={`kpi ${card.tone ?? ''}`} key={card.label} style={{ animationDelay: `${index * 70}ms` }}>
            <div className="kpi-top">
              <span className="kpi-label">{card.label}</span>
              <Icon />
            </div>
            <div className="kpi-value">
              {card.value}
              {card.unit && <span className="kpi-unit">{card.unit}</span>}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/**
 * The maps the answer was measured from.
 *
 * Present only when the tool actually rendered them. An empty placeholder tile would
 * imply the system produced evidence it did not.
 */
export function EvidenceGrid({ result }: { result: ToolResult }) {
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

  return (
    <div className="evidence-grid">
      {tiles.map((tile, index) => (
        <figure className="etile" key={tile.key} style={{ animationDelay: `${index * 80}ms` }}>
          <div className="etile-head">{tile.title}</div>
          <img src={`/api/images/evidence/${tile.file}`} alt={tile.title} />
          <figcaption>{tile.note}</figcaption>
        </figure>
      ))}
    </div>
  )
}
