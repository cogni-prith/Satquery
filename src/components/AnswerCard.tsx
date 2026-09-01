import type { ToolResult } from '../lib/api'
import { humanise } from '../lib/answers'

/**
 * The answer, and how much to trust it.
 *
 * Numbers in the text are picked out in a monospace face because they are the claim being
 * made: everything else in the sentence is scaffolding, and a wrong figure should be
 * findable in under a second.
 *
 * The answer is passed through `humanise` first. Discriminative heads answer with a label
 * from a frozen set -- CDVQA emits `30_to_40` -- and rendering that label as English is
 * presentation, not interpretation: it never substitutes a different answer.
 */

/** Wrap every number so CSS can style it, without touching the words around it. */
function highlight(text: string) {
  const parts = text.split(/(\d[\d,]*\.?\d*\s*(?:%|m²|km²|ha|m2)?)/g)
  return parts.map((part, index) =>
    /^\d/.test(part) ? <span className="num" key={index}>{part}</span> : <span key={index}>{part}</span>,
  )
}

function band(value: number): 'high' | 'medium' | 'low' {
  if (value >= 0.75) return 'high'
  if (value >= 0.45) return 'medium'
  return 'low'
}

function ConfidenceRing({ value }: { value: number }) {
  const radius = 32
  const circumference = 2 * Math.PI * radius
  return (
    <div className={`conf-ring ${band(value)}`} title="Agreement between two independent estimates">
      <svg width="76" height="76">
        <circle className="track" cx="38" cy="38" r={radius} />
        <circle
          className="value"
          cx="38"
          cy="38"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value)}
        />
      </svg>
      <div className="label">
        <div>
          <div className="pct">{Math.round(value * 100)}</div>
          <div className="cap">agree</div>
        </div>
      </div>
    </div>
  )
}

export function AnswerCard({ result }: { result: ToolResult }) {
  return (
    <div className="answer-card">
      <div className="answer-body">
        <div className="answer-text">{highlight(humanise(result.answer) ?? '')}</div>
        {result.confidence != null && <ConfidenceRing value={result.confidence} />}
      </div>
      <div className="answer-foot">
        <span className="tool-badge">{result.tool_name}</span>
        <span className="chip">v{result.tool_version}</span>
        <span className="chip">{result.latency_ms.toFixed(0)} ms</span>
        {result.confidence == null && (
          /* Said out loud rather than left blank: a missing confidence means nothing
             corroborated the answer, which is information, not an empty field. */
          <span className="chip" title="No second signal was available to compare against">
            no agreement signal
          </span>
        )}
      </div>
    </div>
  )
}
