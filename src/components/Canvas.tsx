import { useCallback, useEffect, useRef, useState } from 'react'
import type { BoundingBox, UploadedImage } from '../lib/api'
import { IconSwap, IconTarget, IconX } from './Icons'

/**
 * The imagery stage. The subject of the work, so it gets the room.
 *
 * Two dates are shown as one frame with a draggable curtain rather than two thumbnails
 * side by side. Change is the mandatory requirement of this problem statement, and a
 * curtain is how a person actually sees change: the same pixels, the same place on
 * screen, one date wiped over the other. Two images in two boxes forces the viewer to
 * do the registration in their head, which is exactly the work the tool exists to do.
 */
/** Compact area for the legend. Returns an empty string when no scale was available. */
function fmtArea(m2: number | null | undefined): string {
  if (m2 == null) return ''
  if (m2 === 0) return ' 0'
  if (Math.abs(m2) >= 1_000_000) return ` ${(m2 / 1_000_000).toFixed(2)} km²`
  if (Math.abs(m2) >= 10_000) return ` ${(m2 / 10_000).toFixed(1)} ha`
  return ` ${Math.round(m2).toLocaleString()} m²`
}

export function Canvas({
  images,
  roles,
  boxesFor,
  highlight,
  gainedLost,
  onRemove,
  onSwap,
}: {
  images: UploadedImage[]
  roles: [string, string] | null
  boxesFor: (index: number) => BoundingBox[]
  /** Filename of the transparent layer marking the pixels the answer is about. */
  highlight: string | null
  /** Ground area gained and lost for the highlighted class, in m2, when the GSD is known. */
  gainedLost: { klass: string; gained: number | null; lost: number | null } | null
  onRemove: (id: string) => void
  onSwap: () => void
}) {
  const [split, setSplit] = useState(50)
  const [showHighlight, setShowHighlight] = useState(true)
  const [dragging, setDragging] = useState(false)
  const frame = useRef<HTMLDivElement>(null)

  const moveTo = useCallback((clientX: number) => {
    const box = frame.current?.getBoundingClientRect()
    if (!box) return
    setSplit(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)))
  }, [])

  // Listeners live on the window while dragging so the curtain keeps tracking when the
  // pointer leaves the frame -- otherwise it sticks at the edge, which feels broken.
  useEffect(() => {
    if (!dragging) return
    const move = (event: PointerEvent) => moveTo(event.clientX)
    const up = () => setDragging(false)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [dragging, moveTo])

  if (images.length === 0) return null

  const pair = images.length === 2
  // The frame takes the imagery's own aspect ratio. With a fixed square frame and
  // object-fit:contain, a non-square raster is letterboxed -- and a box positioned as a
  // percentage of the FRAME then sits in the wrong place by exactly the letterbox.
  const ratio =
    images[0].width && images[0].height ? images[0].width / images[0].height : 1

  /**
   * Boxes arrive as ABSOLUTE PIXELS on the source raster -- that is the contract, because
   * VRSBench scores grounding on absolute boxes -- so they are divided by that image's own
   * dimensions here. An earlier version multiplied by 100 as though they were normalised,
   * which turned a 77 px box on a 512 px tile into a width of 7680% and drew nothing
   * visible at all.
   */
  const overlay = (index: number) => {
    const source = images[index]
    const pixelWidth = source?.width ?? 0
    const pixelHeight = source?.height ?? 0
    if (!pixelWidth || !pixelHeight) return null

    return boxesFor(index).map((box, boxIndex) => (
      <div
        className="bbox"
        key={boxIndex}
        style={{
          left: `${(box.x_min / pixelWidth) * 100}%`,
          top: `${(box.y_min / pixelHeight) * 100}%`,
          width: `${((box.x_max - box.x_min) / pixelWidth) * 100}%`,
          height: `${((box.y_max - box.y_min) / pixelHeight) * 100}%`,
          animationDelay: `${boxIndex * 80}ms`,
        }}
      >
        <span>
          {box.label.length > 28 ? `${box.label.slice(0, 28)}…` : box.label}
          {box.score != null && ` ${(box.score * 100).toFixed(0)}%`}
        </span>
      </div>
    ))
  }

  return (
    <div className="stage">
      <div
        className={`frame ${pair ? 'is-pair' : ''}`}
        ref={frame}
        style={{ aspectRatio: String(ratio) }}
      >
        <img className="layer" src={`/api/images/${images[0].image_id}/preview`} alt={images[0].filename} />
        <div className="layer-boxes">{overlay(0)}</div>

        {pair && (
          <>
            {/* The second date is clipped to the curtain. Both layers are absolutely
                positioned on the same grid, so the wipe compares like with like. */}
            <div className="layer-clip" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
              <img className="layer" src={`/api/images/${images[1].image_id}/preview`} alt={images[1].filename} />
              <div className="layer-boxes">{overlay(1)}</div>
            </div>

            <div
              className={`curtain ${dragging ? 'grabbing' : ''}`}
              style={{ left: `${split}%` }}
              onPointerDown={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              role="separator"
              aria-label="Drag to compare dates"
              aria-valuenow={Math.round(split)}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'ArrowLeft') setSplit((v) => Math.max(0, v - 4))
                if (event.key === 'ArrowRight') setSplit((v) => Math.min(100, v + 4))
              }}
            >
              <span className="curtain-grip">
                <i />
                <i />
              </span>
            </div>

            <span className="date-tag left">{roles?.[0] ?? 'date 1'}</span>
            <span className="date-tag right">{roles?.[1] ?? 'date 2'}</span>
          </>
        )}

        {/* The highlight sits above both dates and outside the curtain clip, so the marked
            pixels stay in place while the wipe moves underneath them -- which is what makes
            it read as "this area changed" rather than as part of either image. */}
        {highlight && showHighlight && (
          <img className="layer highlight" src={`/api/images/evidence/${highlight}`} alt="" />
        )}

        {/* The legend has to carry the outline, not just the fill. A red blob alone
            cannot say what it was measured against: on a reservoir that only filled, the
            changed region IS most of the final lake, and the overlay reads as "you have
            drawn the water" until the old shoreline is named. */}
        {highlight && (
          <div className="frame-legend">
            {/* Gained/lost is a claim only the spectral path can make. The radiometric
                fallback marks where the scene differs and cannot say in which direction,
                so on a pair with no class it gets the one label it has earned. */}
            {pair && gainedLost ? (
              <>
                {gainedLost.klass && <span className="key klass">{gainedLost.klass.replace(/_/g, ' ')}</span>}
                <span className="key"><i className="sw gained" /> gained{fmtArea(gainedLost.gained)}</span>
                {/* The zero is printed rather than the row hidden. A scene where nothing
                    was lost is a finding; a legend that quietly drops the category looks
                    like the overlay failed to draw it. */}
                <span className={`key ${gainedLost.lost === 0 ? 'nil' : ''}`}>
                  <i className="sw lost" /> lost{fmtArea(gainedLost.lost)}
                </span>
                <span className="key"><i className="sw rim" /> extent at date 1</span>
              </>
            ) : (
              <span className="key">
                <i className="sw gained" /> {pair ? 'changed appearance' : 'measured area'}
              </span>
            )}
          </div>
        )}

        <div className="frame-tools">
          {highlight && (
            <button
              className={`ghost-btn ${showHighlight ? 'on' : ''}`}
              onClick={() => setShowHighlight((v) => !v)}
              title="Toggle the highlight over the imagery"
            >
              <IconTarget /> {showHighlight ? 'hide' : 'show'} highlight
            </button>
          )}
          {pair && (
            <button className="ghost-btn" onClick={onSwap} title="Swap which image is the earlier date">
              <IconSwap /> swap dates
            </button>
          )}
          {images.map((image, index) => (
            <button
              key={image.image_id}
              className="ghost-btn"
              onClick={() => onRemove(image.image_id)}
              title={`Remove ${image.filename}`}
            >
              <IconX /> {pair ? `date ${index + 1}` : 'clear'}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
