import type { BoundingBox, UploadedImage } from '../lib/api'
import { IconX } from './Icons'

/**
 * One uploaded raster: the preview, what the system read off it, and any boxes.
 *
 * The preview comes from the backend's own `load_model_input`, so a SAR scene is shown
 * through exactly the pipeline the tools see rather than a separate display path. What
 * you look at is what the model looked at.
 */
export function ImageCard({
  image,
  index,
  role,
  boxes,
  onRemove,
}: {
  image: UploadedImage
  index: number
  role: string | null
  boxes: BoundingBox[]
  onRemove: () => void
}) {
  return (
    <div className="img-card" style={{ animationDelay: `${index * 80}ms` }}>
      <div className="img-frame">
        <img src={`/api/images/${image.image_id}/preview`} alt={image.filename} />
        {role && <span className="role-tag">{role}</span>}
        <button className="img-remove" onClick={onRemove} title="Remove" aria-label="Remove image">
          <IconX />
        </button>
        {boxes.map((box, boxIndex) => (
          /* Boxes arrive in normalised coordinates, so they lay over the preview
             correctly whatever its display size. */
          <div
            className="bbox"
            key={boxIndex}
            style={{
              left: `${box.x_min * 100}%`,
              top: `${box.y_min * 100}%`,
              width: `${(box.x_max - box.x_min) * 100}%`,
              height: `${(box.y_max - box.y_min) * 100}%`,
              animationDelay: `${boxIndex * 90}ms`,
            }}
          >
            <span>
              {box.label}
              {box.score != null && ` ${(box.score * 100).toFixed(0)}%`}
            </span>
          </div>
        ))}
      </div>

      <div className="img-meta">
        <div className="img-name" title={image.filename}>
          {image.filename}
        </div>
        <div className="img-chips">
          <span className="chip">{image.modality}</span>
          <span className="chip" title={image.gsd_m ? 'Read from the affine transform' : 'No transform: the GSD is genuinely unknown'}>
            {image.gsd_token}
          </span>
          {image.width && (
            <span className="chip">
              {image.width}×{image.height}
            </span>
          )}
          {image.crs && <span className="chip">{image.crs}</span>}
          {image.band_names.slice(0, 5).map((band) => (
            <span className="chip" key={band}>
              {band}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
