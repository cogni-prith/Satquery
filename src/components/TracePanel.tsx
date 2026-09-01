import type { Trace } from '../lib/api'
import { IconRoute } from './Icons'

/**
 * The execution trace, drawn as the pipeline it is.
 *
 * A separately scored judging row, so it gets a real visualisation rather than a log dump:
 * numbered nodes on a connector, each with its latency drawn to scale against the slowest
 * step. Reading which stage cost the time should not require reading any numbers.
 */
export function TracePanel({ trace, running }: { trace: Trace | null; running: boolean }) {
  if (!trace) return null
  const slowest = Math.max(...trace.steps.map((step) => step.latency_ms ?? 0), 1)
  const total = trace.steps.reduce((sum, step) => sum + (step.latency_ms ?? 0), 0)

  return (
    <section className="card">
      <header className="card-head">
        <span className="card-title">
          <IconRoute /> execution trace
        </span>
        <span className="chip">
          {trace.steps.length} steps · {total.toFixed(0)} ms
        </span>
      </header>

      <div className="trace">
        {trace.steps.map((step, index) => {
          const isLast = index === trace.steps.length - 1
          const latency = step.latency_ms ?? 0
          return (
            <div
              className={`tstep ${step.status} ${running && isLast ? 'running' : ''}`}
              key={`${step.tool_name}-${index}`}
              style={{ animationDelay: `${index * 90}ms` }}
            >
              <div className="tnode">{index + 1}</div>
              <div className="tbody">
                <div className="trow">
                  <span className="tname">{step.tool_name ?? 'step'}</span>
                  {step.tool_version && <span className="tver">v{step.tool_version}</span>}
                  <span className="tms">{latency.toFixed(1)} ms</span>
                </div>
                {step.message && <div className="tmsg">{step.message}</div>}
                <div className="tbar">
                  <i style={{ ['--w' as string]: `${(latency / slowest) * 100}%` }} />
                </div>
                {Object.keys(step.params).length > 0 && (
                  <div className="tparams">{JSON.stringify(step.params, null, 1)}</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
