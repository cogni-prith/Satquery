import type { ToolResult, UploadedImage } from '../lib/api'

/**
 * The bottom strip: the technical read of what is loaded and what just ran.
 *
 * Borrowed from GIS and editing tools, where the truth about the current document lives
 * in a permanent strip rather than a panel you open. GSD and CRS belong here because they
 * silently determine whether every area on screen is right, and they should never need
 * to be gone looking for.
 */
export function StatusStrip({
  images,
  result,
  queued,
}: {
  images: UploadedImage[]
  result: ToolResult | null
  queued: boolean
}) {
  const primary = images[0]

  return (
    <footer className="strip">
      <div className="strip-group">
        {primary ? (
          <>
            <Item label="gsd" value={primary.gsd_token.replace(/[<>]|gsd:/g, '')} accent={primary.gsd_m != null} />
            <Item label="crs" value={primary.crs ?? 'none'} />
            <Item label="bands" value={primary.band_names.join(' ')} mono />
            <Item label="size" value={`${primary.width}×${primary.height}`} />
          </>
        ) : (
          <span className="strip-idle">no imagery loaded</span>
        )}
      </div>

      <div className="strip-group right">
        {queued && <span className="strip-idle live">running…</span>}
        {result && (
          <>
            <Item label="tool" value={`${result.tool_name} v${result.tool_version}`} mono />
            <Item label="latency" value={`${result.latency_ms.toFixed(0)} ms`} />
            <Item
              label="confidence"
              value={result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '—'}
            />
          </>
        )}
      </div>
    </footer>
  )
}

function Item({ label, value, mono, accent }: { label: string; value: string; mono?: boolean; accent?: boolean }) {
  return (
    <span className="strip-item">
      <em>{label}</em>
      <b className={`${mono ? 'mono' : ''} ${accent ? 'accent' : ''}`}>{value}</b>
    </span>
  )
}
