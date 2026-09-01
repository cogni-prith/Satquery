/**
 * Inline SVG icons.
 *
 * Inline rather than an icon package: the whole set is nine glyphs, they inherit
 * `currentColor` so they theme for free, and adding a dependency that ships hundreds of
 * icons to use nine of them is the kind of weight that never comes back off.
 */
const base = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  viewBox: '0 0 24 24',
}

export const IconUpload = () => (
  <svg {...base}><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" /><path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" /></svg>
)
export const IconLayers = () => (
  <svg {...base}><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></svg>
)
export const IconSpark = () => (
  <svg {...base}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" /></svg>
)
export const IconRoute = () => (
  <svg {...base}><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="18" r="2.5" /><path d="M8.5 6H14a4 4 0 0 1 0 8H9a4 4 0 0 0 0 8h.5" /></svg>
)
export const IconWarn = () => (
  <svg {...base}><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9 2.4 17.6A2 2 0 0 0 4.1 20.6h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /></svg>
)
export const IconX = () => (
  <svg {...base}><path d="M18 6 6 18M6 6l12 12" /></svg>
)
export const IconSwap = () => (
  <svg {...base}><path d="M7 4 3 8l4 4" /><path d="M3 8h13a4 4 0 0 1 0 8h-1" /></svg>
)
export const IconTable = () => (
  <svg {...base}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 10v10" /></svg>
)
export const IconGlobe = () => (
  <svg {...base}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18Z" /></svg>
)
