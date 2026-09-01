import { useCallback, useEffect, useRef, useState } from 'react'
import { api, pollJob, type Health, type JobView, type ToolSpec, type UploadedImage } from './lib/api'
import { examplesFor, planFor } from './lib/roles'
import { Canvas } from './components/Canvas'
import { SourceRail } from './components/SourceRail'
import { StatusStrip } from './components/StatusStrip'
import { AnswerCard } from './components/AnswerCard'
import { RecordPanel } from './components/RecordPanel'
import { TracePanel } from './components/TracePanel'
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
        {/* ── top chrome: identity, the command bar, system state ─────────── */}
        <header className="bar">
          <div className="brand">
            <div className="orbit">
              <span className="ring" />
              <span className="sat" />
              <span className="planet" />
            </div>
            <span className="wordmark">SatQuery</span>
          </div>

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
              {busy ? <span className="spinner" /> : <>Run <kbd>↵</kbd></>}
            </button>
          </div>

          <div className="sysbar">
            <span className={`pill ${ready ? 'pill-ok' : health?.detail ? 'pill-bad' : 'pill-wait'}`}>
              <span className="dot" />
              {ready ? 'ready' : health?.detail ? 'failed' : 'starting'}
            </span>
            {ready && health?.vram_used_mb != null && (
              <span className="stat">
                <b>{(health.vram_used_mb / 1024).toFixed(1)}</b> GB
              </span>
            )}
          </div>
        </header>

        {/* ── body: rail · canvas · dock ──────────────────────────────────── */}
        <div className="body">
          <SourceRail
            images={images}
            tools={tools}
            dragging={dragging}
            onPick={() => fileInput.current?.click()}
            onDrop={addFiles}
            onDragState={setDragging}
          />

          <main
            className={`canvas ${dragging ? 'dropping' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              addFiles(event.dataTransfer.files)
            }}
          >
            {images.length === 0 ? (
              <div className="canvas-empty">
                <div className="empty-mark">
                  <IconUpload />
                </div>
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
          </main>

          <aside className="dock">
            {error && (
              <div className="note note-err">
                <IconWarn />
                <div>{error}</div>
              </div>
            )}

            {busy && !result && (
              <div className="dock-block">
                <div className="skel" style={{ height: 22, width: '78%' }} />
                <div className="skel" style={{ height: 22, width: '52%' }} />
                <div className="skel" style={{ height: 54 }} />
              </div>
            )}

            {result?.answer && <AnswerCard result={result} />}

            {jobError && (
              <div className="note note-err">
                <IconWarn />
                <div>{jobError}</div>
              </div>
            )}

            {result?.answer_record && <RecordPanel record={result.answer_record} />}

            {warnings.length > 0 && (
              <section className="dock-section">
                <header className="dock-head">
                  caveats <span className="chip">{warnings.length}</span>
                </header>
                <div className="note-list">
                  {warnings.map((warning, index) => (
                    <div className="note note-warn" key={index} style={{ animationDelay: `${index * 55}ms` }}>
                      <IconWarn />
                      <div>{warning}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <TracePanel trace={job?.trace ?? null} running={busy} />

            {!job && !busy && !error && (
              /* An idle dock that explains the pipeline rather than apologising for being
                 empty. It is also the clearest statement of what makes this system
                 different: the answer is computed, then worded -- not generated. */
              <div className="dock-idle">
                <div className="dock-head">how an answer is produced</div>
                <ol className="pipeline-preview">
                  <li>
                    <b>Route</b>
                    <span>The gate reads modality, band names and GSD, then picks a tool. No model guesses this.</span>
                  </li>
                  <li>
                    <b>Measure</b>
                    <span>Specialists produce masks and boxes. Areas are pixel counts times the GSD from the transform.</span>
                  </li>
                  <li>
                    <b>Record</b>
                    <span>Every value is stored with the tool that produced it. A value with no provenance is refused.</span>
                  </li>
                  <li>
                    <b>Word</b>
                    <span>The verbalizer receives the record and never the image, so it cannot describe what was not measured.</span>
                  </li>
                </ol>
              </div>
            )}
          </aside>
        </div>

        <StatusStrip images={images} result={result} queued={busy} />
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
