import type { Transition, Variants } from 'framer-motion'

/**
 * Shared motion vocabulary.
 *
 * Defined once so every panel, chip and number moves with the same physics. A dozen
 * components each picking their own duration is what makes an interface feel assembled
 * rather than designed.
 *
 * Springs rather than durations for anything that travels: a spring settles at a speed
 * proportional to the distance it covers, so a tall panel and a small chip both arrive
 * feeling like the same material. Durations are kept for fades, which have no distance.
 */

export const spring: Transition = { type: 'spring', stiffness: 320, damping: 32, mass: 0.9 }
export const springSoft: Transition = { type: 'spring', stiffness: 210, damping: 30, mass: 1 }
export const springSnap: Transition = { type: 'spring', stiffness: 520, damping: 34, mass: 0.7 }

export const ease = [0.22, 1, 0.36, 1] as const

/** Panels entering the console: up and in, never sideways. */
export const panelIn: Variants = {
  hidden: { opacity: 0, y: 14, scale: 0.985 },
  show: { opacity: 1, y: 0, scale: 1, transition: spring },
}

/** Wraps a group whose children should arrive one after another. */
export const stagger = (delay = 0.05, initial = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: delay, delayChildren: initial } },
})

/** A row in a list that streams in as data arrives. */
export const rowIn: Variants = {
  hidden: { opacity: 0, x: -8 },
  show: { opacity: 1, x: 0, transition: springSnap },
}

/** Chips, badges, small marks. */
export const popIn: Variants = {
  hidden: { opacity: 0, scale: 0.86 },
  show: { opacity: 1, scale: 1, transition: springSnap },
}

export const fade: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.35, ease } },
  exit: { opacity: 0, transition: { duration: 0.22, ease } },
}

/** Applied to every button so tap feedback is uniform. */
export const tap = { scale: 0.97 }
export const hoverLift = { y: -2 }
