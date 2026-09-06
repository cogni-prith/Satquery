import { motion } from 'framer-motion'
import type { Plan } from '../lib/roles'
import type { Trace, UploadedImage } from '../lib/api'
import { ease, rowIn, spring, stagger } from '../lib/motion'

/**
 * The routing diagram, lit up along the path actually taken.
 *
 * A static "how it works" picture is marketing. This one reads the current images and the
 * returned trace and highlights the branch the deterministic gate really chose, so the
 * diagram is an assertion the system has to keep rather than a drawing beside it.
 *
 * The gate is the honest thing to show here: routing happens by reading modality, band
 * names and GSD, with no model involved. That is why the answer to "which tool ran" is
 * always checkable.
 *
 * The connectors draw themselves down the chosen branch when a route resolves. That is
 * the one animation in this interface carrying an argument rather than a mood: the whole
 * claim of the architecture is that this path is decided, not guessed, and watching it
 * travel is what makes that legible in the seconds a judge gives it.
 */
export function RouteMap({
  images,
  plan,
  trace,
}: {
  images: UploadedImage[]
  plan: Plan
  trace: Trace | null
}) {
  const count = images.length
  const chosen =
    trace?.steps.find((step) => step.tool_name === 'router.select')?.params?.chosen ?? null

  const branch = count === 0 ? null : count === 1 ? 'single' : 'pair'
  const leaves = [
    { key: 'vlm.vqa', label: 'Visual Q&A', side: 'single' },
    { key: 'vlm.caption', label: 'Captioning', side: 'single' },
    { key: 'indices.deterministic', label: 'Measure & compare', side: 'pair' },
    { key: 'fusion.extraction', label: 'Optical + SAR', side: 'pair' },
  ]

  /** A connector draws top-down when its branch is live, and stays a hairline when not. */
  const line = (live: boolean, delay: number) => (
    <motion.span
      className={`rline ${live ? 'lit' : ''}`}
      initial={false}
      animate={{ scaleY: 1, opacity: 1 }}
      transition={{ duration: 0.45, delay: live ? delay : 0, ease }}
    />
  )

  return (
    <div className="routemap">
      <motion.div
        className={`rnode root ${count ? 'lit' : ''}`}
        animate={count ? { scale: [1, 1.02, 1] } : {}}
        transition={{ duration: 0.5, ease }}
      >
        <em>query</em>
        <b>What is needed?</b>
      </motion.div>

      <div className="rsplit">
        {line(branch === 'single', 0.05)}
        {line(branch === 'pair', 0.05)}
      </div>

      <div className="rrow">
        <motion.div className={`rnode ${branch === 'single' ? 'lit' : ''}`} layout transition={spring}>
          <em>1 image</em>
          <b>Single scene</b>
        </motion.div>
        <motion.div className={`rnode ${branch === 'pair' ? 'lit' : ''}`} layout transition={spring}>
          <em>2 images</em>
          <b>{plan.config === 'cross_modal_pair' ? 'Cross-modal pair' : 'Bi-temporal pair'}</b>
        </motion.div>
      </div>

      <div className="rsplit four">
        {leaves.map((leaf) => (
          <span key={leaf.key}>{line(branch === leaf.side, 0.2)}</span>
        ))}
      </div>

      <div className="rrow four">
        {leaves.map((leaf) => (
          <motion.div
            className={`rleaf ${chosen === leaf.key ? 'chosen' : branch === leaf.side ? 'lit' : ''}`}
            key={leaf.key}
            layout
            transition={spring}
            animate={
              chosen === leaf.key
                ? { scale: [1, 1.06, 1], transition: { duration: 0.55, delay: 0.3, ease } }
                : {}
            }
          >
            {leaf.label}
            {chosen === leaf.key && (
              <motion.span
                className="rtick"
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.45, ...spring }}
              >
                selected
              </motion.span>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  )
}

/**
 * What the gate decided, as plain rows.
 *
 * Deliberately not styled as a model's "thoughts": every row here is a value read off the
 * imagery or a lookup in the registry, and presenting deterministic routing as reasoning
 * would oversell it.
 */
export function DecisionPanel({
  images,
  plan,
  trace,
  busy,
}: {
  images: UploadedImage[]
  plan: Plan
  trace: Trace | null
  busy: boolean
}) {
  const select = trace?.steps.find((step) => step.tool_name === 'router.select')
  const chosen = (select?.params?.chosen as string) ?? null
  const modality = images[0]?.modality ?? null

  const rows: [string, string | null, boolean?][] = [
    ['Images loaded', images.length ? String(images.length) : null],
    ['Modality', modality],
    ['Ground sample', images[0] ? images[0].gsd_token.replace(/[<>]|gsd:/g, '') : null],
    ['Input config', images.length ? plan.config.replace(/_/g, ' ') : null],
    ['Tool selected', chosen, true],
  ]

  return (
    <motion.div className="decision" variants={stagger(0.05)} initial="hidden" animate="show">
      {rows.map(([label, value, highlight]) => (
        <motion.div className="drow" key={label} variants={rowIn}>
          <span className="dlabel">{label}</span>
          <span className={`dvalue ${highlight && value ? 'hot' : ''} ${value ? '' : 'none'}`}>
            {value ?? '—'}
          </span>
        </motion.div>
      ))}
      {/* A trace with no tool chosen means the gate refused: that is a decision, not an
          idle state, and reporting it as "awaiting a query" is how the interface came to
          look like it had simply ignored the request. */}
      <motion.div className={`dstatus ${busy ? 'busy' : !chosen && trace ? 'refused' : ''}`} variants={rowIn}>
        <span className="dpulse" />
        {busy
          ? 'Routing and measuring…'
          : chosen
            ? 'Completed'
            : trace
              ? 'Refused — no tool accepts this input'
              : 'Awaiting a query'}
      </motion.div>
    </motion.div>
  )
}
