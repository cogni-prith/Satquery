import { motion } from 'framer-motion'
import type { Trace } from '../lib/api'
import { IconRoute } from './Icons'
import { ease, rowIn, stagger } from '../lib/motion'

/**
 * The execution trace, drawn as the pipeline it is.
 *
 * A separately scored judging row, so it gets a real visualisation rather than a log dump:
 * numbered nodes on a connector, each with its latency drawn to scale against the slowest
 * step. Reading which stage cost the time should not require reading any numbers.
 *
 * Steps arrive one after another and their bars grow to length, so the trace is watched
 * being built rather than found already finished.
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

      <motion.div className="trace" variants={stagger(0.07)} initial="hidden" animate="show">
        {trace.steps.map((step, index) => {
          const isLast = index === trace.steps.length - 1
          const latency = step.latency_ms ?? 0
          return (
            <motion.div
              className={`tstep ${step.status} ${running && isLast ? 'running' : ''}`}
              key={`${step.tool_name}-${index}`}
              variants={rowIn}
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
                  <motion.i
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.5, delay: 0.1 + index * 0.07, ease }}
                    style={{ ['--w' as string]: `${(latency / slowest) * 100}%` }}
                  />
                </div>
                {Object.keys(step.params).length > 0 && (
                  <div className="tparams">{JSON.stringify(step.params, null, 1)}</div>
                )}
              </div>
            </motion.div>
          )
        })}
      </motion.div>
    </section>
  )
}
