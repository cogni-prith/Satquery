import type { ToolSpec, UploadedImage } from '../lib/api'
import { IconUpload } from './Icons'

/**
 * The left rail: what is loaded, and what the system can do with it.
 *
 * Narrow and dense on purpose. This is reference material the analyst glances at, not
 * content they read, so it sits at the edge of vision and never competes with the canvas.
 */
export function SourceRail({
  images,
  tools,
  dragging,
  onPick,
  onDrop,
  onDragState,
}: {
  images: UploadedImage[]
  tools: ToolSpec[]
  dragging: boolean
  onPick: () => void
  onDrop: (files: FileList) => void
  onDragState: (over: boolean) => void
}) {
  return (
    <aside className="rail">
      <div className="rail-section">
        <div className="rail-label">sources</div>

        {images.map((image, index) => (
          <div className="source" key={image.image_id} style={{ animationDelay: `${index * 60}ms` }}>
            <img src={`/api/images/${image.image_id}/preview`} alt="" />
            <div className="source-meta">
              <div className="source-name" title={image.filename}>
                {image.filename}
              </div>
              <div className="source-sub">
                {image.modality} · {image.width}×{image.height}
              </div>
            </div>
          </div>
        ))}

        {images.length < 2 && (
          <button
            className={`rail-drop ${dragging ? 'over' : ''}`}
            onClick={onPick}
            onDragOver={(event) => {
              event.preventDefault()
              onDragState(true)
            }}
            onDragLeave={() => onDragState(false)}
            onDrop={(event) => {
              event.preventDefault()
              onDragState(false)
              onDrop(event.dataTransfer.files)
            }}
          >
            <IconUpload />
            <span>{images.length === 0 ? 'add imagery' : 'add second date'}</span>
          </button>
        )}
      </div>

      <div className="rail-section">
        <div className="rail-label">
          tools
          <span className="rail-count">
            {tools.filter((tool) => tool.implemented).length}/{tools.length}
          </span>
        </div>
        {tools.map((tool) => (
          <div className={`tool-row ${tool.implemented ? 'on' : 'off'}`} key={tool.name} title={tool.description}>
            <span className="sig" />
            <span className="tool-name">{tool.name}</span>
          </div>
        ))}
      </div>
    </aside>
  )
}
