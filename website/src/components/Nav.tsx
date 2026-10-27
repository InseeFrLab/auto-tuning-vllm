import { motion, useReducedMotion } from 'motion/react'
import { DOCS_URL, GITHUB_URL } from '../links'

const linkClass =
  'text-muted-ink text-sm tracking-wide transition-colors duration-300 hover:text-off-white'

export function Nav() {
  const reducedMotion = useReducedMotion()

  return (
    <motion.header
      className="flex shrink-0 items-center justify-between pt-[clamp(1.25rem,3vh,2rem)]"
      initial={reducedMotion ? false : { opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
    >
      <span className="font-display text-[clamp(1rem,1.6vw,1.125rem)] font-medium tracking-tight text-off-white">
        Auto-Tune vLLM
      </span>
      <nav className="flex items-center gap-[clamp(1.25rem,3vw,2.5rem)]">
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className={linkClass}
        >
          Documentation
        </a>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className={linkClass}
        >
          GitHub
        </a>
      </nav>
    </motion.header>
  )
}
