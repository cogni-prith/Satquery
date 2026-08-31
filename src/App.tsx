import { useEffect, useState } from 'react'
import { api, pollJob, type Health, type JobView, type ToolSpec, type UploadedImage } from './lib/api'
import { ResultPanel } from './components/ResultPanel'
import { TracePanel } from './components/TracePanel'

/**
 * SatQuery AI.
 *
 * The UI holds no domain logic. It uploads rasters, submits a query, and renders whatever
 * the contract returns. Which tools exist, what they accept and whether they are
 * implemented all come from `/api/tools`, so there is no second capability list here to
 * drift from the registry the router actually reads.
 */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [tools, setTools] = useState<ToolSpec[]>([])
  const [images, setImages] = useState<UploadedImage[]>([])
  const [query, setQuery] = useState('')
  const [job, setJob] = useState<JobView | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Models take about 45 seconds to load. Poll until they are up so the UI can say
  // "starting" rather than letting a user submit into a 503.
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const next = await api.health()
        if (!cancelled) setHealth(next)
        if (!next.models_loaded && !cancelled) setTimeout(tick, 2000)
      } catch {
        if (!cancelled) setTimeout(tick, 3000)
      }
    }
    tick()
    api.tools().then((body) => !cancelled && setTools(body.tools)).catch(() => {})
    return () => { cancelled = true }
  }, [])

  const onUpload = async (files: FileList | null) => {
    if (!files?.length) return
    setError(null)
    for (const file of Array.from(files).slice(0, 2 - images.length)) {
      try {
        const uploaded = await api.upload(file)
        setImages((current) => [...current, uploaded])
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc))
      }
    }
  }

  const onSubmit = async () => {
    if (!query.trim() || images.length === 0) return
    setBusy(true)
    setError(null)
    setJob(null)
    try {
      const { job_id } = await api.submit(query, images.map((image) => image.image_id))
      await pollJob(job_id, setJob)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(false)
    }
  }

  const ready = health?.models_loaded === true
  const implemented = tools.filter((tool) => tool.implemented)

  return (
    <div className="app">
      <header>
        <h1>SatQuery AI</h1>
        <p className="tagline">
          Vision-language analysis of remote sensing imagery, queried in natural language.
        </p>
        <div className="status">
          <span className={ready ? 'dot dot-ok' : 'dot dot-wait'} />
          {ready ? (
            <>
              models resident
              {health?.vram_used_mb != null && (
                <span className="muted"> · {(health.vram_used_mb / 1024).toFixed(1)} GB VRAM</span>
              )}
            </>
          ) : health?.detail ? (
            <span className="error-text">{health.detail}</span>
          ) : (
            <>loading models — about 45 seconds</>
          )}
          {health && <span className="muted"> · contract v{health.contract_version}</span>}
        </div>
      </header>

      <section className="panel">
        <h2>1 · Imagery</h2>
        <p className="muted">
          One raster for VQA, captioning or grounding. Two for change or optical-plus-SAR
          fusion. GeoTIFF preferred — GSD is read from the affine transform, never guessed.
        </p>
        <input
          type="file"
          accept=".tif,.tiff,.png,.jpg"
          multiple
          disabled={images.length >= 2}
          onChange={(event) => onUpload(event.target.files)}
        />
        {images.length > 0 && (
          <ul className="images">
            {images.map((image) => (
              <li key={image.image_id}>
                <b>{image.filename}</b>
                <span className="muted">
                  {' '}
                  {image.modality} · {image.gsd_token} · {image.width}×{image.height} ·{' '}
                  {image.band_names.join(', ')}
                </span>
                {image.warnings.length > 0 && (
                  <ul className="warnings">
                    {image.warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        )}
        {images.length > 0 && (
          <button className="link" onClick={() => { setImages([]); setJob(null) }}>
            clear
          </button>
        )}
      </section>

      <section className="panel">
        <h2>2 · Question</h2>
        <textarea
          value={query}
          placeholder="How many aircraft are in this image?"
          onChange={(event) => setQuery(event.target.value)}
          rows={3}
        />
        <button onClick={onSubmit} disabled={busy || !ready || !query.trim() || !images.length}>
          {busy ? (job?.status ?? 'working') : ready ? 'Ask' : 'models loading'}
        </button>
        {error && <p className="error-text">{error}</p>}
      </section>

      <ResultPanel result={job?.result ?? null} />
      <TracePanel trace={job?.trace ?? null} />

      <section className="panel">
        <h2>Available tools</h2>
        <p className="muted">Read from the registry — the same source the router reads.</p>
        <ul className="tools">
          {implemented.map((tool) => (
            <li key={tool.name}>
              <code>{tool.name}</code> <span className="muted">{tool.description}</span>
            </li>
          ))}
        </ul>
        {tools.length > implemented.length && (
          <p className="muted">
            {tools.length - implemented.length} further tools are registered but not
            implemented. They fail honestly rather than returning a fabricated answer.
          </p>
        )}
      </section>
    </div>
  )
}
