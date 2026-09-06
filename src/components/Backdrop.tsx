/**
 * The backdrop behind the console: drifting aurora blooms over a fixed starfield.
 *
 * Pure CSS animation on a handful of elements rather than a canvas render loop. A loop
 * would run a repaint every frame for the whole session to draw something nobody looks
 * at directly, and it would compete with the analysis request for the main thread at
 * exactly the moment the interface needs to feel quick. These transforms and opacities
 * are composited off the main thread instead.
 *
 * Stars are generated once at module scope, not per render, so they do not reshuffle
 * every time a result arrives.
 */

const STAR_COUNT = 70

/** Deterministic, so the field is identical across reloads and screenshots compare. */
function seeded(seed: number): () => number {
  let state = seed
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296
    return state / 4294967296
  }
}

const STARS = (() => {
  const random = seeded(20260906)
  return Array.from({ length: STAR_COUNT }, () => ({
    left: random() * 100,
    top: random() * 100,
    size: 0.8 + random() * 1.6,
    delay: random() * 6,
    duration: 3.5 + random() * 4,
    opacity: 0.25 + random() * 0.5,
  }))
})()

export function Backdrop() {
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="stars">
        {STARS.map((star, index) => (
          <span
            key={index}
            style={{
              left: `${star.left}%`,
              top: `${star.top}%`,
              width: star.size,
              height: star.size,
              opacity: star.opacity,
              animationDelay: `${star.delay}s`,
              animationDuration: `${star.duration}s`,
            }}
          />
        ))}
      </div>
      <div className="bloom bloom-a" />
      <div className="bloom bloom-b" />
      <div className="bloom bloom-c" />
      <div className="grid-veil" />
      <div className="vignette" />
    </div>
  )
}
