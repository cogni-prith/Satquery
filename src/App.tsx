import { useCallback, useEffect, useRef, useState } from 'react'
import { api, pollJob, type Health, type JobView, type ToolSpec, type UploadedImage } from './lib/api'
import { examplesFor, planFor } from './lib/roles'
import { Canvas } from './components/Canvas'
import { Sidebar } from './components/Sidebar'
import { StatusStrip } from './components/StatusStrip'
import { AnswerCard } from './components/AnswerCard'
import { RecordPanel } from './components/RecordPanel'
import { TracePanel } from './components/TracePanel'
import { DecisionPanel, RouteMap } from './components/RouteMap'
import { EvidenceGrid, KpiRow } from './components/Results'
import { IconGlobe, IconSpark, IconUpload, IconWarn } from './components/Icons'

/**
 * SatQuery AI.
 *
 * Laid out as an analysis tool, not a dashboard: a narrow source rail, the imagery filling
 * the middle, evidence docked at the right, and a permanent status strip along the bottom.
 * The imagery is the subject of the work and gets the space; everything else is chrome
 * pushed to the edges.
 *
 * The query sits in the top bar rather than in a panel because it is the one control used
 * on every single interaction, and burying the primary action inside a card is how a tool
 * ends up feeling like a form.
 *
 * The UI holds no domain logic. Capabilities come from `/api/tools`, the same registry the
 * router reads, so there is no second list here to drift from the real one.
 */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [tools, setTools] = useState<ToolSpec[]>([])
  const [images, setImages] = useState<UploadedImage[]>([])
  const [query, setQuery] = useState('')
  const [job, setJob] = useState<JobView | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [demoBusy, setDemoBusy] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const plan = planFor(images)
  const examples = examplesFor(plan)
  const ready = health?.models_loaded === true

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const next = await api.health()
        if (cancelled) return
        setHealth(next)
        if (!next.models_loaded) setTimeout(tick, 2000)
      } catch {
        if (!cancelled) setTimeout(tick, 3000)
      }
    }
    tick()
    api.tools().then((body) => !cancelled && setTools(body.tools)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const addFiles = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files) return
      setError(null)
      const room = 2 - images.length
      for (const file of Array.from(files).slice(0, room)) {
        try {
          const uploaded = await api.upload(file)
          setImages((current) => (current.length >= 2 ? current : [...current, uploaded]))
        } catch (exc) {
          setError(exc instanceof Error ? exc.message : String(exc))
        }
      }
    },
    [images.length],
  )

  const loadDemo = async () => {
    setDemoBusy(true)
    setError(null)
    try {
      setImages(await api.demo())
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setDemoBusy(false)
    }
  }

  const onSubmit = async () => {
    if (!query.trim() || images.length === 0 || busy) return
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

  const result = job?.result ?? null
  const warnings = result?.warnings ?? []
  const jobError = job?.error ?? result?.error ?? null
  const canRun = ready && !busy && query.trim().length > 0 && images.length > 0

  return (
    <>
      <div className="aurora" />

      <div className="app">
        <Sidebar tools={tools} active={busy ? 'change' : null} />

        <div className="main">
          <header className="bar">
            <div className={`command ${busy ? 'busy' : ''}`}>
              <IconSpark />
              <input
                value={query}
                placeholder={
                  images.length === 0
                    ? 'Load imagery to begin…'
                    : plan.config === 'bi_temporal_pair'
                      ? 'Ask what changed between the two dates…'
                      : 'Ask about extent, objects, or description…'
                }
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    onSubmit()
                  }
                }}
                disabled={images.length === 0}
              />
              <button className="command-run" disabled={!canRun} onClick={onSubmit}>
                {busy ? <span className="spinner" /> : <>Analyse <kbd>↵</kbd></>}
              </button>
            </div>

            <div className="sysbar">
              <span className={`pill ${ready ? 'pill-ok' : health?.detail ? 'pill-bad' : 'pill-wait'}`}>
                <span className="dot" />
                {ready ? 'system online' : health?.detail ? 'failed' : 'starting'}
              </span>
              {ready && health?.vram_used_mb != null && (
                <span className="stat"><b>{(health.vram_used_mb / 1024).toFixed(1)}</b> GB</span>
              )}
            </div>
          </header>

          <div className="scroll">
            {/* ── how it routes, and what it decided ─────────────────────────── */}
            <section className="panel-row">
              <section className="panel">
                <header className="panel-head">
                  <h2>How SatQuery routes a query</h2>
                  <span className="chip">deterministic gate · no model</span>
                </header>
                <div className="panel-body">
                  <RouteMap images={images} plan={plan} trace={job?.trace ?? null} />
                </div>
              </section>

              <section className="panel">
                <header className="panel-head">
                  <h2>Router decision</h2>
                </header>
                <div className="panel-body">
                  <DecisionPanel images={images} plan={plan} trace={job?.trace ?? null} busy={busy} />
                </div>
              </section>
            </section>

            {/* ── ask ───────────────────────────────────────────────────────── */}
            <section className="panel">
              <header className="panel-head">
                <h2>Imagery</h2>
                <div className="head-tools">
                  <span className="chip">{images.length}/2 loaded</span>
                  {images.length > 0 && (
                    <button className="ghost-btn" onClick={() => setImages([])}>clear</button>
                  )}
                </div>
              </header>

              <div
                className={`panel-body stage-body ${dragging ? 'dropping' : ''}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragging(false)
                  addFiles(event.dataTransfer.files)
                }}
              >
                {images.length === 0 ? (
                  <div className="canvas-empty">
                    <div className="empty-mark"><IconUpload /></div>
                    <h2>Drop a scene to begin</h2>
                    <p>
                      One raster for extent, objects or description. Two acquisitions of the same
                      place for change — they arrive as a single frame you wipe between.
                    </p>
                    <div className="empty-actions">
                      <button className="solid-btn" onClick={loadDemo} disabled={demoBusy}>
                        {demoBusy ? <span className="spinner" /> : <IconGlobe />}
                        Load demo scene
                      </button>
                      <button className="ghost-btn lg" onClick={() => fileInput.current?.click()}>
                        Browse files
                      </button>
                    </div>
                    <div className="formats">
                      <span className="chip">GeoTIFF</span>
                      <span className="chip">multispectral</span>
                      <span className="chip">SAR</span>
                      <span className="chip">PNG / JPEG</span>
                    </div>
                  </div>
                ) : (
                  <>
                    <Canvas
                      images={images}
                      roles={plan.roles}
                      boxesFor={(index) =>
                        (result?.evidence.boxes ?? []).filter((box) => box.image_index === index)
                      }
                      highlight={result?.evidence.highlight_path?.split('/').pop() ?? null}
                      gainedLost={(() => {
                        const klass = result?.params_used?.highlighted_class as string | undefined
                        if (!klass) return null
                        const t = result?.answer_record?.change_transitions ?? {}
                        // Absent when the imagery carried no scale: the split is real but
                        // has no ground area, so the legend shows the colours without figures.
                        const gained = t[`to_${klass}`]
                        const lost = t[`from_${klass}`]
                        return {
                          klass,
                          gained: typeof gained === 'number' ? gained : null,
                          lost: typeof lost === 'number' ? lost : null,
                        }
                      })()}
                      onRemove={(id) => setImages((current) => current.filter((item) => item.image_id !== id))}
                      onSwap={() => setImages((current) => [current[1], current[0]])}
                    />
                    <div className="canvas-foot">
                      <div className="plan-inline">
                        <IconGlobe />
                        <b>{plan.headline}</b>
                        <span>{plan.detail}</span>
                      </div>
                      {examples.length > 0 && !busy && (
                        <div className="examples">
                          {examples.slice(0, 3).map((example) => (
                            <button className="example" key={example} onClick={() => setQuery(example)}>
                              {example}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </section>

            {error && (
              <div className="note note-err"><IconWarn /><div>{error}</div></div>
            )}

            {/* A refused query returns a completed job with an error and no result, so it
                cannot live inside the result panel -- gating that panel on `result` is
                exactly how a refusal became silence. It gets its own block, above the
                fold, because "why did nothing happen" is the question it answers. */}
            {!busy && jobError && !result && (
              <section className="panel refusal">
                <header className="panel-head">
                  <h2>Cannot answer this</h2>
                  <span className="chip">no tool ran</span>
                </header>
                <div className="panel-body refusal-body">
                  <IconWarn />
                  <p>{jobError}</p>
                </div>
              </section>
            )}

            {/* ── result ────────────────────────────────────────────────────── */}
            {(busy || result) && (
              <section className="panel accent">
                <header className="panel-head">
                  <h2>Analysis result</h2>
                  {result && <span className="chip">{result.tool_name} · {result.latency_ms.toFixed(0)} ms</span>}
                </header>

                <div className="panel-body">
                  {busy && !result ? (
                    <div className="dock-block">
                      <div className="skel" style={{ height: 24, width: '70%' }} />
                      <div className="skel" style={{ height: 68 }} />
                      <div className="skel" style={{ height: 120 }} />
                    </div>
                  ) : result ? (
                    <>
                      {result.answer_record && <KpiRow record={result.answer_record} result={result} />}
                      <div className="result-split">
                        <div className="result-main">
                          {result.answer && <AnswerCard result={result} />}
                          {jobError && (
                            <div className="note note-err"><IconWarn /><div>{jobError}</div></div>
                          )}
                          <EvidenceGrid result={result} />
                        </div>
                        <div className="result-side">
                          {result.answer_record && <RecordPanel record={result.answer_record} />}
                          {warnings.length > 0 && (
                            <section className="card">
                              <header className="card-head">
                                <span className="card-title"><IconWarn /> caveats</span>
                                <span className="chip">{warnings.length}</span>
                              </header>
                              <div className="note-list">
                                {warnings.map((warning, index) => (
                                  <div className="note note-warn" key={index} style={{ animationDelay: `${index * 55}ms` }}>
                                    <IconWarn /><div>{warning}</div>
                                  </div>
                                ))}
                              </div>
                            </section>
                          )}
                          <TracePanel trace={job?.trace ?? null} running={busy} />
                        </div>
                      </div>
                    </>
                  ) : null}
                </div>
              </section>
            )}

            <footer className="foot">
              SatQuery AI · SIH 2026 · every figure on this page is measured, and names the tool that measured it
            </footer>
          </div>

          <StatusStrip images={images} result={result} queued={busy} />
        </div>
      </div>

      <input
        ref={fileInput}
        type="file"
        hidden
        multiple
        accept=".tif,.tiff,.png,.jpg,.jpeg"
        onChange={(event) => {
          addFiles(event.target.files)
          event.target.value = ''
        }}
      />
    </>
  )
}
