import type { ToolSpec } from '../lib/api'
import { IconCaption, IconChange, IconFusion, IconQuestion } from './Icons'

/**
 * The left sidebar: what the system can do, and what is actually live.
 *
 * Every capability is listed, including the ones that are not ready. Hiding an untrained
 * model would make the sidebar look better and the system look dishonest -- and the gap
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
    <aside className="side">
      <div className="side-brand">
        <div className="orbit">
          <span className="ring" />
          <span className="sat" />
          <span className="planet" />
        </div>
        <div>
          <div className="wordmark">
            SatQuery <span>AI</span>
          </div>
          <div className="side-sub">Remote sensing intelligence</div>
        </div>
      </div>

      <nav className="side-group">
        <div className="side-label">capabilities</div>
        {CAPABILITIES.map((capability) => {
          const live = capability.tools.some((tool) => bound.has(tool))
          const Icon = capability.icon
          return (
            <div
              className={`nav-item ${live ? '' : 'dim'} ${active === capability.key ? 'active' : ''}`}
              key={capability.key}
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
            </div>
          )
        })}
      </nav>

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
    </aside>
  )
}
