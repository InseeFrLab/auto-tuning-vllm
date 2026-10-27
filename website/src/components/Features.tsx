import { motion, useReducedMotion } from 'motion/react'

const FEATURES = [
  'Workload-aware tuning',
  'Multi-objective optimization',
  'GuideLLM benchmarks',
  'Speculative decoding',
] as const

export function Features() {
  const reducedMotion = useReducedMotion()

  return (
    <motion.footer
      className="shrink-0 pb-[clamp(1.25rem,3vh,2rem)] pt-[clamp(0.75rem,2vh,1.5rem)]"
      initial={reducedMotion ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.7,
        delay: 0.35,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <ul className="grid grid-cols-2 gap-x-6 gap-y-3 lg:flex lg:flex-wrap lg:items-center lg:justify-center lg:gap-x-[clamp(2rem,5vw,4.5rem)] lg:gap-y-2">
        {FEATURES.map((label, index) => (
          <motion.li
            key={label}
            className="text-center text-[clamp(0.7rem,1.1vw,0.8125rem)] tracking-wide text-muted-ink lg:text-center"
            initial={reducedMotion ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.55,
              delay: 0.45 + index * 0.06,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            {label}
          </motion.li>
        ))}
      </ul>
    </motion.footer>
  )
}
