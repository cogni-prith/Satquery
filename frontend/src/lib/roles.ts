import type { UploadedImage } from './api'

/**
 * What the router will do with the images currently loaded.
 *
 * This mirrors `app/router/gate.py` on the client so the UI can say what is about to
 * happen *before* the user commits to a query. It is a preview, never an authority: the
 * backend's gate decides, and this file exists only so the interface is not silent about
 * a decision the system has already effectively made.
 *
 * Keeping the two in step is a real maintenance cost. It is worth paying because the
 * alternative -- a user uploading two rasters with no idea whether they are about to get
 * change detection or fusion -- is the interface hiding the most interesting thing the
 * system does.
 */

export type Plan = {
  config: 'single' | 'cross_modal_pair' | 'bi_temporal_pair' | 'none'
  headline: string
  detail: string
  roles: [string, string] | null
  /** True when the pair's order changes the meaning, so the UI must offer a swap. */
  ordered: boolean
}

export function planFor(images: UploadedImage[]): Plan {
  if (images.length === 0) {
    return {
      config: 'none',
      headline: 'No imagery yet',
      detail: 'Add one raster for question answering, captioning or grounding. Add two for change detection or optical-plus-SAR fusion.',
      roles: null,
      ordered: false,
    }
  }

  if (images.length === 1) {
    return {
      config: 'single',
      headline: 'Single image',
      detail: 'Question answering, captioning and grounding are reachable. The router picks one from your question.',
      roles: null,
      ordered: false,
    }
  }

  const [a, b] = images
  const sarCount = images.filter((image) => image.modality === 'sar').length

  // Sensor difference is tested before timestamp difference, matching the backend. An
  // optical/SAR pair of one scene usually differs in both, because they are separate
  // acquisitions -- checking timestamps first would call every fusion pair a change pair.
  if (sarCount === 1) {
    return {
      config: 'cross_modal_pair',
      headline: 'Optical + SAR pair',
      detail: 'Joint built-up and water extraction. The learned model is cross-checked against the deterministic indices, and their agreement is the reported confidence.',
      roles: a.modality === 'sar' ? ['SAR', 'Optical'] : ['Optical', 'SAR'],
      ordered: false, // the backend orders by modality, not by upload position
    }
  }

  if (sarCount === 2) {
    return {
      config: 'bi_temporal_pair',
      headline: 'Two SAR images',
      detail: 'Treated as a before/after pair. Order matters — the earlier acquisition must come first.',
      roles: ['Before', 'After'],
      ordered: true,
    }
  }

  const dated = a.gsd_m !== null && b.gsd_m !== null
  return {
    config: 'bi_temporal_pair',
    headline: 'Bi-temporal pair',
    detail: dated
      ? 'Change detection over the closed CDVQA answer set. Order matters — the earlier acquisition must come first.'
      : 'Change detection. Neither raster carries an acquisition timestamp, so the ordering below is the one that will be used.',
    roles: ['Before', 'After'],
    ordered: true,
  }
}

/** Example questions that reach each configuration, for the query chips. */
export function examplesFor(plan: Plan): string[] {
  switch (plan.config) {
    case 'single':
      return [
        'Describe this scene',
        'How many buildings are visible?',
        'Where is the water?',
        'What colour are the large vehicles?',
      ]
    case 'cross_modal_pair':
      return ['Extract built-up areas and water', 'Where is the water in this pair?']
    case 'bi_temporal_pair':
      return [
        'Did the areas of buildings change?',
        'What changed between these images?',
        'Did water increase or decrease?',
      ]
    default:
      return []
  }
}
