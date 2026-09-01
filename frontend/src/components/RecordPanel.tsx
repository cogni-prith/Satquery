import type { AnswerRecord } from '../lib/api'
import { IconTable } from './Icons'

/**
 * The measured record behind the sentence.
 *
 * This panel is the reason to prefer this system over one that generates an answer: every
 * figure is shown beside the tool that produced it, so a judge can check the claim instead
 * of trusting it. A generative model cannot fill this in, which is the point.
 */

const UNIT_LABEL: Record<string, string> = { m2: 'm²', fraction: '', count: '' }

/** Square metres get carried into hectares and km² where that is the readable unit. */
function formatValue(value: unknown, unit: string | null): { text: string; unit: string } {
  if (typeof value !== 'number') return { text: String(value), unit: unit ?? '' }

  if (unit === 'm2') {
    if (Math.abs(value) >= 1_000_000) return { text: (value / 1_000_000).toFixed(2), unit: 'km²' }
    if (Math.abs(value) >= 10_000) return { text: (value / 10_000).toFixed(2), unit: 'ha' }
    return { text: value.toLocaleString(undefined, { maximumFractionDigits: 0 }), unit: 'm²' }
  }
  if (unit === 'fraction') return { text: (value * 100).toFixed(1), unit: '%' }
  if (unit === 'count') return { text: String(value), unit: '' }
  return {
    text: value.toLocaleString(undefined, { maximumFractionDigits: 3 }),
    unit: UNIT_LABEL[unit ?? ''] ?? unit ?? '',
  }
}

function humanise(key: string) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function Facts({ record }: { record: AnswerRecord }) {
  if (record.facts.length === 0) return null
  return (
    <div className="facts">
      {record.facts.map((fact, index) => {
        const { text, unit } = formatValue(fact.value, fact.unit)
        return (
          <div className="fact" key={fact.key} style={{ animationDelay: `${index * 55}ms` }}>
            <div className="fact-key">{humanise(fact.key)}</div>
            <div className="fact-val">
              {text}
              {unit && <span className="unit">{unit}</span>}
            </div>
            <div className="fact-prov" title="The tool that produced this value">
              {fact.provenance}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Deltas({ record }: { record: AnswerRecord }) {
  const entries = Object.entries(record.area_deltas)
  if (entries.length === 0) return null

  return (
    <>
      {entries.map(([name, delta], index) => {
        // Without a ground scale the bars show share of the scene instead of hectares.
        // Same shape, weaker claim, and the unit on the row says which one you are reading.
        const scaled = delta.absolute_m2 != null
        const t1 = scaled ? (delta.area_t1_m2 ?? 0) : (delta.fraction_t1 ?? 0)
        const t2 = scaled ? (delta.area_t2_m2 ?? 0) : (delta.fraction_t2 ?? 0)
        const peak = Math.max(t1, t2, scaled ? 1 : 0.0001)
        const rows: [string, number][] = [
          ['T1', t1],
          ['T2', t2],
        ]
        return (
          <div className="delta" key={name} style={{ animationDelay: `${index * 70}ms` }}>
            <div className="delta-head">
              <span className="delta-name">{humanise(name)}</span>
              <span className={`trend ${delta.trend}`}>
                {delta.trend === 'increased' ? '▲' : delta.trend === 'decreased' ? '▼' : '—'}
                {delta.trend}
              </span>
            </div>
            <div className="delta-bars">
              {rows.map(([label, area]) => {
                const { text, unit } = formatValue(area, scaled ? 'm2' : 'fraction')
                return (
                  <div className="dbar" key={label}>
                    <span className="t">{label}</span>
                    <span className="track">
                      <i className="fill" style={{ ['--w' as string]: `${(area / peak) * 100}%` }} />
                    </span>
                    <span className="v">
                      {text} {unit}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </>
  )
}

function Proportions({ record }: { record: AnswerRecord }) {
  const entries = Object.entries(record.class_proportions).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return null
  return (
    <div className="props">
      {entries.map(([name, value]) => (
        <div className={`prop ${name}`} key={name}>
          <span className="prop-name">{name.replace(/_/g, ' ')}</span>
          <span className="track">
            <i className="fill" style={{ ['--w' as string]: `${Math.min(value * 100, 100)}%` }} />
          </span>
          <span className="pct">{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export function RecordPanel({ record }: { record: AnswerRecord }) {
  const hasContent =
    record.facts.length > 0 ||
    Object.keys(record.area_deltas).length > 0 ||
    Object.keys(record.class_proportions).length > 0

  if (!hasContent) return null

  return (
    <section className="card">
      <header className="card-head">
        <span className="card-title">
          <IconTable /> measured record
        </span>
        <span className="chip">{record.intent}</span>
      </header>
      <Proportions record={record} />
      <Deltas record={record} />
      <Facts record={record} />
    </section>
  )
}
