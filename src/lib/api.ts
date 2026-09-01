/**
 * The backend contract, as the frontend sees it.
 *
 * These types mirror `satquery.serve.contracts` deliberately loosely: the backend
 * serialises the ML layer's own Pydantic models straight to the wire, so anything this
 * file over-specifies is a second definition of a contract it does not own. Fields the UI
 * renders are typed; the rest passes through.
 */

export type ToolSpec = {
  name: string
  version: string
  task: string
  description: string
  implemented: boolean
  accepted_input_configs: string[]
  accepted_modalities: string[]
  requires_gpu: boolean
}

export type UploadedImage = {
  image_id: string
  filename: string
  modality: string
  gsd_m: number | null
  gsd_token: string
  width: number | null
  height: number | null
  band_names: string[]
  crs: string | null
  warnings: string[]
}

export type BoundingBox = {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
  label: string
  score: number | null
  image_index: number
}

/** One measured value plus the tool that produced it. The provenance is never blank. */
export type Fact = {
  key: string
  value: unknown
  unit: string | null
  provenance: string
  confidence: number | null
}

export type AreaDelta = {
  area_t1_m2: number
  area_t2_m2: number
  absolute_m2: number
  relative: number
  trend: 'increased' | 'decreased' | 'unchanged'
}

/**
 * The record the answer was computed from.
 *
 * This is the part of the system worth showing. The text answer is a rendering of this
 * object, so every number on screen can be traced back to the tool that measured it --
 * which is the difference between an answer and a plausible sentence.
 */
export type AnswerRecord = {
  intent: string
  facts: Fact[]
  class_proportions: Record<string, number>
  object_counts: Record<string, number>
  area_deltas: Record<string, AreaDelta>
  change_transitions: Record<string, number>
  referenced_regions: Record<string, unknown>[]
  agreement_scores: Record<string, number>
  warnings: string[]
}

export type ToolResult = {
  request_id: string
  tool_name: string
  tool_version: string
  answer: string | null
  confidence: number | null
  evidence: {
    boxes: BoundingBox[]
    mask_path: string | null
    overlay_path: string | null
    highlight_path: string | null
    index_maps: Record<string, string>
  }
  answer_record: AnswerRecord | null
  params_used: Record<string, unknown>
  latency_ms: number
  warnings: string[]
  error: string | null
}

export type TraceStep = {
  step: string
  tool_name: string | null
  tool_version: string | null
  params: Record<string, unknown>
  latency_ms: number | null
  confidence: number | null
  status: 'ok' | 'warning' | 'error' | 'skipped'
  message: string | null
}

export type Trace = {
  request_id: string
  input_config: string | null
  task: string | null
  steps: TraceStep[]
  total_latency_ms?: number | null
}

export type JobView = {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  result: ToolResult | null
  trace: Trace | null
  error: string | null
}

export type Health = {
  models_loaded: boolean
  loading: boolean
  vram_used_mb: number | null
  contract_version: string
  detail: string | null
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // FastAPI puts the reason in `detail`. Surfacing it verbatim matters: the backend's
    // errors name what is missing (a model, a band, a file), and paraphrasing them in the
    // UI would throw away the only actionable part.
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* response had no JSON body; the status line is all we have */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(unwrap<Health>),

  tools: () =>
    fetch('/api/tools').then(unwrap<{ contract_version: string; tools: ToolSpec[] }>),

  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return fetch('/api/images', { method: 'POST', body: form }).then(unwrap<UploadedImage>)
  },

  /** Ingest the bundled demo scene. 404s when none is installed, which is not an error. */
  demo: () => fetch('/api/images/demo', { method: 'POST' }).then(unwrap<UploadedImage[]>),

  submit: (query: string, imageIds: string[]) =>
    fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, image_ids: imageIds }),
    }).then(unwrap<{ job_id: string }>),

  job: (jobId: string) => fetch(`/api/jobs/${jobId}`).then(unwrap<JobView>),
}

/**
 * Poll a job to completion.
 *
 * Polling rather than a websocket: at 3.5 second latencies the difference is invisible to
 * a user, and a dropped socket on a laptop's wifi mid-demo is a failure mode with no
 * upside here.
 */
export async function pollJob(
  jobId: string,
  onUpdate: (job: JobView) => void,
  intervalMs = 600,
): Promise<JobView> {
  for (;;) {
    const job = await api.job(jobId)
    onUpdate(job)
    if (job.status === 'done' || job.status === 'failed') return job
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
}
