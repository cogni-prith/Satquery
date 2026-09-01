import type { Plan } from '../lib/roles'
import type { Trace, UploadedImage } from '../lib/api'

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

  return (
    <div className="routemap">
      <div className={`rnode root ${count ? 'lit' : ''}`}>
        <em>query</em>
        <b>What is needed?</b>
      </div>

      <div className="rsplit">
        <span className={`rline ${branch === 'single' ? 'lit' : ''}`} />
        <span className={`rline ${branch === 'pair' ? 'lit' : ''}`} />
      </div>

      <div className="rrow">
        <div className={`rnode ${branch === 'single' ? 'lit' : ''}`}>
          <em>1 image</em>
          <b>Single scene</b>
        </div>
        <div className={`rnode ${branch === 'pair' ? 'lit' : ''}`}>
          <em>2 images</em>
          <b>{plan.config === 'cross_modal_pair' ? 'Cross-modal pair' : 'Bi-temporal pair'}</b>
        </div>
      </div>

      <div className="rsplit four">
        {leaves.map((leaf) => (
          <span key={leaf.key} className={`rline ${branch === leaf.side ? 'lit' : ''}`} />
        ))}
      </div>

      <div className="rrow four">
        {leaves.map((leaf) => (
          <div
            className={`rleaf ${chosen === leaf.key ? 'chosen' : branch === leaf.side ? 'lit' : ''}`}
            key={leaf.key}
          >
            {leaf.label}
            {chosen === leaf.key && <span className="rtick">selected</span>}
          </div>
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
    <div className="decision">
      {rows.map(([label, value, highlight]) => (
        <div className="drow" key={label}>
          <span className="dlabel">{label}</span>
          <span className={`dvalue ${highlight && value ? 'hot' : ''} ${value ? '' : 'none'}`}>
            {value ?? '—'}
          </span>
        </div>
      ))}
      <div className={`dstatus ${busy ? 'busy' : ''}`}>
        <span className="dpulse" />
        {busy ? 'Routing and measuring…' : chosen ? 'Completed' : 'Awaiting a query'}
      </div>
    </div>
  )
}
