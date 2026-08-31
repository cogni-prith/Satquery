import type { ToolResult } from '../lib/api'

/**
 * The answer, its confidence, and the evidence behind it.
 *
 * Two rules this panel exists to honour:
 *
 * A tool that errored still returns a well-formed result, and that is rendered as an
 * error, never as an empty answer. The ML layer refuses to fabricate; the UI must not
 * quietly present a blank where a refusal happened.
 *
 * Confidence, where fusion supplies it, is agreement between the learned model and a
 * deterministic spectral index -- not a softmax. It is labelled as such, because a number
 * called "confidence" that means two different things in two places is worse than no
 * number at all.
 */
export function ResultPanel({ result }: { result: ToolResult | null }) {
  if (!result) return null

  if (result.error) {
    return (
      <section className="panel panel-error">
        <h2>The tool could not answer</h2>
        <p className="error-text">{result.error}</p>
        <p className="muted">
          This is a refusal, not a failure to load. The system reports what is missing
          rather than returning a plausible answer it cannot support.
        </p>
      </section>
    )
  }

  const agreement = result.params_used?.agreement as Record<string, number> | undefined

  return (
    <section className="panel">
      <h2>Answer</h2>
      <p className="answer">{result.answer ?? '—'}</p>

      <div className="result-meta">
        <span><b>tool</b> {result.tool_name} v{result.tool_version}</span>
        <span><b>latency</b> {result.latency_ms.toFixed(0)} ms</span>
        {result.confidence != null && (
          <span title="Agreement between the learned model and the deterministic index layer, not a softmax.">
            <b>confidence</b> {result.confidence.toFixed(3)} <span className="muted">(index agreement)</span>
          </span>
        )}
      </div>

      {agreement && (
        <div className="agreement">
          {Object.entries(agreement).map(([name, value]) => (
            <span key={name} className={value < 0.5 ? 'agree-low' : 'agree-ok'}>
              {name.replace('_', ' ')} {value.toFixed(2)}
            </span>
          ))}
        </div>
      )}

      {result.evidence.boxes.length > 0 && (
        <>
          <h3>Boxes</h3>
          <ul className="boxes">
            {result.evidence.boxes.map((box, index) => (
              <li key={index}>
                <code>
                  [{box.x_min.toFixed(0)}, {box.y_min.toFixed(0)}, {box.x_max.toFixed(0)},{' '}
                  {box.y_max.toFixed(0)}]
                </code>{' '}
                {box.label}
                {box.score != null && <span className="muted"> · {box.score.toFixed(2)}</span>}
              </li>
            ))}
          </ul>
        </>
      )}

      {(result.evidence.mask_path || Object.keys(result.evidence.index_maps).length > 0) && (
        <>
          <h3>Rasters written</h3>
          <ul className="rasters">
            {result.evidence.mask_path && (
              <li><b>mask</b> <code>{result.evidence.mask_path}</code></li>
            )}
            {Object.entries(result.evidence.index_maps).map(([name, path]) => (
              <li key={name}><b>{name}</b> <code>{path}</code></li>
            ))}
          </ul>
          <p className="muted">
            Written under the artifact root so they can be opened in QGIS alongside the
            original scene.
          </p>
        </>
      )}

      {result.warnings.length > 0 && (
        <>
          <h3>Warnings</h3>
          <ul className="warnings">
            {result.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
