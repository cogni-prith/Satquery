import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, MotionConfig, motion, useReducedMotion } from 'framer-motion'
import { api, pollJob, type Health, type JobView, type ToolSpec, type UploadedImage } from './lib/api'
import { examplesFor, planFor } from './lib/roles'
import { Canvas } from './components/Canvas'
import { Sidebar } from './components/Sidebar'
import { StatusStrip } from './components/StatusStrip'
import { Backdrop } from './components/Backdrop'
import { Intro } from './components/Intro'
import { ResultTray } from './components/ResultTray'
import { RecordPanel } from './components/RecordPanel'
import { TracePanel } from './components/TracePanel'
import { DecisionPanel, RouteMap } from './components/RouteMap'
import { IconGlobe, IconRoute, IconSpark, IconUpload, IconWarn } from './components/Icons'
import { ease, panelIn, popIn, stagger, tap } from './lib/motion'

/**
 * SatQuery AI.
 *
 * A console over a live backdrop: capabilities on a rail that expands when you reach for
 * it, the imagery centre stage, and the router's reasoning docked to the right as
 * separate instruments rather than one wall of panel.
 *
 * The imagery is the subject of the work, so it is the only thing that gets to grow, and
 * nothing is allowed to displace it once it is on screen. That is why the answer arrives
 * on a tray that floats over the foot of the stage instead of a panel below it: at the
 * moment a result lands the operator is looking at the picture, and a layout that shoves
 * the picture to make room costs them their place in it.
 *
 * The UI holds no domain logic. Capabilities come from `/api/tools`, the same registry the
 * router reads, so there is no second list here to drift from the real one. Every figure
 * shown is one the backend measured; the motion here animates how numbers appear, never
 * what they are.
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
  // Once per session, not once per reload: the sequence is worth watching the first time
  // and an obstacle every time after, and a demo gets reloaded a great deal.
  const [intro, setIntro] = useState(() => sessionStorage.getItem('satquery.intro') !== 'done')
  const fileInput = useRef<HTMLInputElement>(null)
  const reduced = useReducedMotion()

  const plan = planFor(images)
  const examples = examplesFor(plan)
  const ready = health?.models_loaded === true

  const endIntro = useCallback(() => {
    sessionStorage.setItem('satquery.intro', 'done')
    setIntro(false)
  }, [])

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
  const jobError = job?.error ?? result?.error ?? null
  const canRun = ready && !busy && query.trim().length > 0 && images.length > 0

  return (
    <MotionConfig reducedMotion={reduced ? 'always' : 'never'}>
      <Backdrop />

      <AnimatePresence>{intro && <Intro key="intro" onDone={endIntro} />}</AnimatePresence>

      <motion.div
        className="app"
        variants={stagger(0.07, intro ? 0.2 : 0)}
        initial="hidden"
        animate="show"
      >
        <Sidebar tools={tools} active={busy ? 'change' : null} />

        <div className="main">
          <motion.header className="bar" variants={panelIn}>
            <div className={`command ${busy ? 'busy' : ''}`}>
              <span className="command-mark"><IconSpark /></span>
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
              <motion.button
                className="command-run"
                disabled={!canRun}
                onClick={onSubmit}
                whileTap={canRun ? tap : undefined}
                whileHover={canRun ? { scale: 1.02 } : undefined}
              >
                {busy ? <span className="spinner" /> : <>Analyse <kbd>↵</kbd></>}
              </motion.button>
              {/* Sweeps once per run: the only place the interface says "working" on the
                  same surface the question was typed into. */}
              {busy && <span className="command-scan" />}
            </div>

            <div className="sysbar">
              <span className={`pill ${ready ? 'pill-ok' : health?.detail ? 'pill-bad' : 'pill-wait'}`}>
                <span className="dot" />
                {ready ? 'system online' : health?.detail ? 'failed' : 'starting'}
              </span>
              {ready && health?.vram_used_mb != null && (
                <motion.span className="stat" variants={popIn}>
                  <b>{(health.vram_used_mb / 1024).toFixed(1)}</b> GB
                </motion.span>
              )}
            </div>
          </motion.header>

          <div className="work">
            {/* ── centre: the imagery, with the answer floating at its foot ──── */}
            <div className="col centre">
              <motion.section className="panel stage-panel" variants={panelIn}>
                <header className="panel-head">
                  <h2>Imagery</h2>
                  <div className="head-tools">
                    <span className="chip">{images.length}/2 loaded</span>
                    {images.length < 2 && (
                      <motion.button
                        className="ghost-btn"
                        whileTap={tap}
                        onClick={() => fileInput.current?.click()}
                      >
                        add
                      </motion.button>
                    )}
                    {images.length > 0 && (
                      <motion.button className="ghost-btn" whileTap={tap} onClick={() => setImages([])}>
                        clear
                      </motion.button>
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
                  <AnimatePresence mode="wait">
                    {images.length === 0 ? (
                      <motion.div
                        className="canvas-empty"
                        key="empty"
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.2 } }}
                      >
                        <div className="empty-mark"><IconUpload /></div>
                        <h2>Drop a scene to begin</h2>
                        <p>
                          One raster for extent, objects or description. Two acquisitions of the same
                          place for change — they arrive as a single frame you wipe between.
                        </p>
                        <div className="empty-actions">
                          <motion.button
                            className="solid-btn"
                            whileTap={tap}
                            whileHover={{ scale: 1.03 }}
                            onClick={loadDemo}
                            disabled={demoBusy}
                          >
                            {demoBusy ? <span className="spinner" /> : <IconGlobe />}
                            Load demo scene
                          </motion.button>
                          <motion.button
                            className="ghost-btn lg"
                            whileTap={tap}
                            onClick={() => fileInput.current?.click()}
                          >
                            Browse files
                          </motion.button>
                        </div>
                        <motion.div className="formats" variants={stagger(0.05)} initial="hidden" animate="show">
                          {['GeoTIFF', 'multispectral', 'SAR', 'PNG / JPEG'].map((label) => (
                            <motion.span className="chip" key={label} variants={popIn}>{label}</motion.span>
                          ))}
                        </motion.div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="canvas"
                        className="stage-mount"
                        initial={{ opacity: 0, scale: 0.985 }}
                        animate={{ opacity: 1, scale: 1, transition: { duration: 0.4, ease } }}
                      >
                        <Canvas
                          images={images}
                          roles={plan.roles}
                          scanning={busy}
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
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {images.length > 0 && (
                  <div className="canvas-foot">
                    <div className="plan-inline">
                      <IconGlobe />
                      <b>{plan.headline}</b>
                      <span>{plan.detail}</span>
                    </div>
                    {examples.length > 0 && !busy && (
                      <motion.div className="examples" variants={stagger(0.06)} initial="hidden" animate="show">
                        {examples.slice(0, 3).map((example) => (
                          <motion.button
                            className="example"
                            key={example}
                            variants={popIn}
                            whileTap={tap}
                            whileHover={{ y: -1 }}
                            onClick={() => setQuery(example)}
                          >
                            {example}
                          </motion.button>
                        ))}
                      </motion.div>
                    )}
                  </div>
                )}
              </motion.section>

              <AnimatePresence>
                {error && (
                  <motion.div
                    className="note note-err"
                    key="err"
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <IconWarn /><div>{error}</div>
                  </motion.div>
                )}

                {/* A refused query returns a completed job with an error and no result, so it
                    cannot live inside the result tray -- gating that tray on `result` is
                    exactly how a refusal became silence. */}
                {!busy && jobError && !result && (
                  <motion.section className="panel refusal" key="refusal" variants={panelIn} initial="hidden" animate="show" exit={{ opacity: 0 }}>
                    <header className="panel-head">
                      <h2>Cannot answer this</h2>
                      <span className="chip">no tool ran</span>
                    </header>
                    <div className="panel-body refusal-body">
                      <IconWarn />
                      <p>{jobError}</p>
                    </div>
                  </motion.section>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {(busy || result) && (
                  <ResultTray key="tray" result={result} busy={busy} jobError={jobError} />
                )}
              </AnimatePresence>
            </div>

            {/* ── dock: why this tool ran, and the numbers behind the sentence ── */}
            <motion.aside className="col dock" variants={stagger(0.08)} initial="hidden" animate="show">
              <motion.section className="panel" variants={panelIn}>
                <header className="panel-head">
                  <h2><IconRoute /> Routing</h2>
                  <span className="chip">deterministic · no model</span>
                </header>
                <div className="panel-body">
                  <RouteMap images={images} plan={plan} trace={job?.trace ?? null} />
                  <DecisionPanel images={images} plan={plan} trace={job?.trace ?? null} busy={busy} />
                </div>
              </motion.section>

              <AnimatePresence>
                {result?.answer_record && (
                  <motion.div key="record" variants={panelIn} initial="hidden" animate="show" exit={{ opacity: 0, y: 10 }}>
                    <RecordPanel record={result.answer_record} />
                  </motion.div>
                )}
              </AnimatePresence>

              <motion.div variants={panelIn}>
                <TracePanel trace={job?.trace ?? null} running={busy} />
              </motion.div>

              <motion.footer className="foot" variants={panelIn}>
                SatQuery AI · SIH 2026 · every figure on this page is measured, and names the
                tool that measured it
              </motion.footer>
            </motion.aside>
          </div>

          <StatusStrip images={images} result={result} queued={busy} />
        </div>
      </motion.div>

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
    </MotionConfig>
  )
}
