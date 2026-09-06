import { motion, useReducedMotion } from 'framer-motion'
import { useEffect } from 'react'
import { ease } from '../lib/motion'

/**
 * The opening sequence.
 *
 * A mark draws itself, the name arrives, the overlay lifts away to reveal the console
 * already assembled behind it. It runs once per session and any key or click ends it
 * early, because the second time you see an animation you are waiting for it, and at a
 * judging booth the same person watches the app start several times.
 *
 * Nothing here loads or blocks: the console mounts underneath at the same moment, so the
 * sequence costs the demo no time it would not have spent on the first request anyway.
 */

const HOLD_MS = 2100

export function Intro({ onDone }: { onDone: () => void }) {
  const reduced = useReducedMotion()

  useEffect(() => {
    const finish = () => onDone()
    const timer = setTimeout(finish, reduced ? 0 : HOLD_MS)
    window.addEventListener('keydown', finish)
    window.addEventListener('pointerdown', finish)
    return () => {
      clearTimeout(timer)
      window.removeEventListener('keydown', finish)
      window.removeEventListener('pointerdown', finish)
    }
  }, [onDone, reduced])

  return (
    <motion.div
      className="intro"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, filter: 'blur(12px)', transition: { duration: 0.5, ease } }}
    >
      <motion.div
        className="intro-inner"
        exit={{ y: -28, transition: { duration: 0.5, ease } }}
      >
        <svg className="intro-mark" viewBox="0 0 120 120" fill="none">
          <motion.circle
            cx="60" cy="60" r="30"
            stroke="var(--accent)" strokeWidth="1.5"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.9, ease }}
          />
          <motion.ellipse
            cx="60" cy="60" rx="52" ry="20"
            stroke="var(--accent-2)" strokeWidth="1"
            transform="rotate(-24 60 60)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.75 }}
            transition={{ duration: 1.1, delay: 0.18, ease }}
          />
          <motion.path
            d="M32 60a28 28 0 0 1 56 0"
            stroke="var(--water)" strokeWidth="2" strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.45, ease }}
          />
          <motion.circle
            cx="98" cy="38" r="4" fill="var(--accent-2)"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.95, ease }}
          />
        </svg>

        <motion.h1
          className="intro-word"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6, ease }}
        >
          SatQuery <span>AI</span>
        </motion.h1>

        <motion.p
          className="intro-tag"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.85, ease }}
        >
          Measured answers from satellite imagery
        </motion.p>

        <motion.div
          className="intro-rule"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 1, delay: 1, ease }}
        />

        <motion.span
          className="intro-skip"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 1.4 }}
        >
          press any key to skip
        </motion.span>
      </motion.div>
    </motion.div>
  )
}
