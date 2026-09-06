import { motion } from 'framer-motion'
import type { ToolSpec } from '../lib/api'
import { IconCaption, IconChange, IconFusion, IconQuestion } from './Icons'
import { panelIn, rowIn, stagger } from '../lib/motion'

/**
 * The left rail: what the system can do, and what is actually live.
 *
 * Collapsed to icons and reopened by reaching for it. The capability list is reference
 * material — it is read once at the start and then never again during a run — so it does
 * not deserve a permanent 250px of a screen whose whole point is the imagery. Expanding
 * on hover rather than on a click means the cost of consulting it is nothing.
 *
 * The expand is CSS on `:hover` and `:focus-within` rather than React state, so it also
 * opens for a keyboard user tabbing into it, with no handler to keep in step.
 *
 * Every capability is listed, including the ones that are not ready. Hiding an untrained
 * model would make the rail look better and the system look dishonest -- and the gap
 * between "specified" and "trained" is the real state of the project, so the interface
 * shows it.
 *
 * Status comes from `/api/tools`, the same registry the router reads. There is no second
 * list here to drift out of step with the backend.
 */

type Capability = {
  key: string
  title: string
  blurb: string
  icon: () => JSX.Element
  /** Tools that must be bound for this capability to be usable. */
  tools: string[]
}

const CAPABILITIES: Capability[] = [
  {
    key: 'vqa',
    title: 'Visual Q&A',
    blurb: 'Ask a question about one scene',
    icon: IconQuestion,
    tools: ['vlm.vqa'],
  },
  {
    key: 'caption',
    title: 'Captioning',
    blurb: 'Describe land cover and objects',
    icon: IconCaption,
    tools: ['vlm.caption'],
  },
  {
    key: 'change',
    title: 'Change detection',
    blurb: 'Compare two dates of one place',
    icon: IconChange,
    tools: ['indices.deterministic', 'change.vqa_head'],
  },
  {
    key: 'fusion',
    title: 'Optical + SAR',
    blurb: 'Cross-check radar against optical',
    icon: IconFusion,
    tools: ['indices.deterministic', 'fusion.extraction'],
  },
]

export function Sidebar({ tools, active }: { tools: ToolSpec[]; active: string | null }) {
  const bound = new Set(tools.filter((tool) => tool.implemented).map((tool) => tool.name))

  return (
    <motion.aside className="side" variants={panelIn}>
      <div className="side-brand">
        <div className="orbit">
          <span className="ring" />
          <span className="sat" />
          <span className="planet" />
        </div>
        <div className="side-brand-text">
          <div className="wordmark">
            SatQuery <span>AI</span>
          </div>
          <div className="side-sub">Remote sensing intelligence</div>
        </div>
      </div>

      <motion.nav className="side-group" variants={stagger(0.05)} initial="hidden" animate="show">
        <div className="side-label">capabilities</div>
        {CAPABILITIES.map((capability) => {
          const live = capability.tools.some((tool) => bound.has(tool))
          const Icon = capability.icon
          return (
            <motion.div
              className={`nav-item ${live ? '' : 'dim'} ${active === capability.key ? 'active' : ''}`}
              key={capability.key}
              variants={rowIn}
              title={`${capability.title} — ${capability.blurb}`}
            >
              <span className="nav-icon">
                <Icon />
              </span>
              <span className="nav-text">
                <b>{capability.title}</b>
                <em>{capability.blurb}</em>
              </span>
              {!live && <span className="badge">soon</span>}
              {active === capability.key && <span className="badge live">running</span>}
            </motion.div>
          )
        })}
      </motion.nav>

      <div className="side-group">
        <div className="side-label">
          models <span className="side-count">{bound.size}/{tools.length}</span>
        </div>
        {tools.map((tool) => (
          <div className={`model-row ${tool.implemented ? 'on' : 'off'}`} key={tool.name} title={tool.description}>
            <span className="sig" />
            <span className="model-name">{tool.name}</span>
            <span className={`badge ${tool.implemented ? 'ok' : ''}`}>
              {tool.implemented ? 'active' : 'untrained'}
            </span>
          </div>
        ))}
      </div>

      <div className="side-about">
        <div className="about-glow" />
        <h4>About</h4>
        <p>
          Perception is discriminative and answers are computed, not generated. The language
          model never sees the image — it only words a record of what was measured.
        </p>
      </div>
    </motion.aside>
  )
}
