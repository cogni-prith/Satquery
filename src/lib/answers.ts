import type { ToolResult } from './api'

/**
 * Making a tool's raw output legible without changing what it says.
 *
 * `change.vqa_head` is a classifier over a frozen 19-value answer set, so it emits label
 * strings like `30_to_40` rather than prose. That is deliberate -- the project routes
 * closed-set change questions to a discriminative head because it beats a generative model
 * on accuracy and answers in milliseconds -- but a bare label is not an answer a person can
 * read. These helpers render the label; they never substitute a different one.
 */

/** Human-readable rendering of a frozen CDVQA label. Unknown labels pass through. */
export function humanise(answer: string | null): string | null {
  if (!answer) return answer

  const range = answer.match(/^(\d+)_to_(\d+)$/)
  if (range) return `${range[1]}–${range[2]}% of the scene`
  if (answer === '0') return 'none of the scene'

  const names: Record<string, string> = {
    NVG_surface: 'non-vegetated ground surface',
    low_vegetation: 'low vegetation',
    yes: 'Yes',
    no: 'No',
  }
  return names[answer] ?? answer.replace(/_/g, ' ')
}

export type ConfidenceKind = {
  /** What this number actually is. */
  label: string
  /** Whether it is the project's principled agreement signal or a bare model score. */
  principled: boolean
  explain: string
}

/**
 * What a confidence number means, per tool.
 *
 * This distinction is not pedantry. `fusion.extraction` reports agreement between a
 * learned model and a closed-form spectral index -- two independent estimates, where
 * disagreement is genuine evidence of uncertainty. `change.vqa_head` reports a softmax
 * over its answer set, which is a measure of how peaked the network's output is and says
 * nothing about whether it is right. On out-of-distribution imagery a classifier will
 * happily return 0.90 for a wrong answer.
 *
 * Presenting both under one word "confidence" would be the exact failure the project
 * warns about elsewhere: a number that means two different things in two places is worse
 * than no number at all.
 */
export function confidenceKind(result: ToolResult): ConfidenceKind {
  if (result.tool_name === 'fusion.extraction') {
    return {
      label: 'learned-vs-index agreement',
      principled: true,
      explain:
        'Intersection over union between the learned mask and the deterministic spectral index. Two independent estimates, so disagreement is real evidence of uncertainty.',
    }
  }
  if (result.tool_name === 'change.vqa_head') {
    return {
      label: 'softmax over the closed answer set',
      principled: false,
      explain:
        'How peaked the classifier’s output is — not a calibrated probability of being correct. A network far outside its training distribution can be confidently wrong.',
    }
  }
  return { label: 'model score', principled: false, explain: 'A raw model score, not a calibrated probability.' }
}

/**
 * Warn when a tool is being run well outside the data it was trained on.
 *
 * The change head was trained on SECOND: 0.53 m aerial imagery of urban scenes. Handed a
 * 10 m Sentinel patch, or any raster whose GSD could not even be computed, its answer is
 * an extrapolation. Saying so costs nothing and is the difference between a demo that
 * overclaims and one a judge can trust.
 */
export function distributionCaution(result: ToolResult, gsds: (number | null)[]): string | null {
  if (result.tool_name !== 'change.vqa_head') return null

  if (gsds.some((gsd) => gsd === null)) {
    return 'This head was trained on 0.53 m aerial imagery (SECOND). The ground sampling distance of this input could not be computed, so there is no way to check whether it is comparable — treat the answer as an extrapolation.'
  }
  const coarse = gsds.filter((gsd): gsd is number => gsd !== null).some((gsd) => gsd > 2)
  return coarse
    ? 'This head was trained on 0.53 m aerial imagery (SECOND). This input is an order of magnitude coarser, so the answer is an extrapolation rather than a measurement.'
    : null
}
