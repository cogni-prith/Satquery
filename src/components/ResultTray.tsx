import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import type { ToolResult } from '../lib/api'
import { AnswerCard } from './AnswerCard'
import { EvidenceGrid, KpiRow } from './Results'
import { IconWarn } from './Icons'
import { ease, panelIn, spring, stagger, tap } from '../lib/motion'

/**
 * The answer, on a tray that rises over the foot of the stage.
 *
 * It floats rather than sitting in the column flow so that arriving at an answer never
 * moves the imagery. The picture is what the operator is looking at when the result
 * lands; a panel that appears above it and shoves it up costs them their place.
 *
 * Collapsible because the evidence tiles are worth real height, and once they have been
 * read the imagery should get the room back without clearing the result.
 */
export function ResultTray({
  result,
  busy,
  jobError,
}: {
  result: ToolResult | null
  busy: boolean
  jobError: string | null
}) {
  const [open, setOpen] = useState(true)

  return (
    <motion.section
      className="tray"
      variants={panelIn}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: 20, transition: { duration: 0.25, ease } }}
      layout
    >
      <header className="tray-head" onClick={() => setOpen((value) => !value)}>
        <span className="tray-grip" />
        <h2>
          {busy && !result ? 'Analysing' : 'Analysis result'}
          {busy && !result && <span className="dots"><i /><i /><i /></span>}
        </h2>
        <div className="head-tools">
          {result && (
            <motion.span className="chip mono" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              {result.tool_name} · {result.latency_ms.toFixed(0)} ms
            </motion.span>
          )}
          <motion.button
            className="ghost-btn"
            whileTap={tap}
            onClick={(event) => {
              event.stopPropagation()
              setOpen((value) => !value)
            }}
          >
            {open ? 'collapse' : 'expand'}
          </motion.button>
        </div>
      </header>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            className="tray-body"
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1, transition: spring }}
            exit={{ height: 0, opacity: 0, transition: { duration: 0.24, ease } }}
          >
            <div className="tray-pad">
              {busy && !result ? (
                <div className="dock-block">
                  <div className="skel" style={{ height: 26, width: '62%' }} />
                  <div className="skel" style={{ height: 70 }} />
                  <div className="skel" style={{ height: 108 }} />
                </div>
              ) : result ? (
                <motion.div variants={stagger(0.08)} initial="hidden" animate="show">
                  {result.answer_record && <KpiRow record={result.answer_record} result={result} />}
                  {result.answer && <AnswerCard result={result} />}
                  {jobError && (
                    <div className="note note-err"><IconWarn /><div>{jobError}</div></div>
                  )}
                  <EvidenceGrid result={result} />
                </motion.div>
              ) : null}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}
