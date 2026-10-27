import { motion, useReducedMotion } from 'motion/react'
import { GITHUB_URL, QUICK_START_URL } from '../links'

const stagger = 0.08

export function Hero() {
  const reducedMotion = useReducedMotion()
  const motionProps = (delay: number) =>
    reducedMotion
      ? {}
      : {
          initial: { opacity: 0, y: 12 } as const,
          animate: { opacity: 1, y: 0 } as const,
          transition: {
            duration: 0.7,
            delay,
            ease: [0.22, 1, 0.36, 1] as const,
          },
        }

  return (
    <div className="max-w-xl lg:max-w-[34rem] lg:pl-[clamp(0rem,2vw,2rem)]">
      <motion.p
        className="font-mono-label mb-[clamp(0.75rem,2vh,1.25rem)] text-[clamp(0.625rem,1vw,0.6875rem)] tracking-[0.2em] text-beige/90"
        {...motionProps(0.05)}
      >
        OPEN SOURCE · LLM INFERENCE
      </motion.p>
      <motion.h1
        className="font-display mb-[clamp(0.5rem,1.5vh,1rem)] text-[clamp(2.75rem,6.5vw,4.5rem)] font-normal leading-[1.05] tracking-tight text-off-white"
        {...motionProps(0.05 + stagger)}
      >
        Auto-Tune vLLM
      </motion.h1>
      <motion.p
        className="font-display mb-[clamp(0.75rem,2vh,1.25rem)] text-[clamp(1.25rem,2.4vw,1.75rem)] leading-snug text-off-white/95"
        {...motionProps(0.05 + stagger * 2)}
      >
        Better inference configurations, found automatically.
      </motion.p>
      <motion.p
        className="mb-[clamp(1.25rem,3vh,2rem)] max-w-md text-[clamp(0.875rem,1.35vw,1rem)] leading-relaxed text-muted-ink"
        {...motionProps(0.05 + stagger * 3)}
      >
        Optimize vLLM for your actual workloads. Explore configurations,
        benchmark performance, and find the right balance between latency and
        throughput.
      </motion.p>
      <motion.div
        className="flex flex-wrap items-center gap-x-8 gap-y-4"
        {...motionProps(0.05 + stagger * 4)}
      >
        <motion.a
          href={QUICK_START_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center rounded-sm bg-off-white px-6 py-2.5 text-sm font-medium tracking-wide text-warm-black"
          whileHover={reducedMotion ? undefined : { y: -1, opacity: 0.92 }}
          whileTap={reducedMotion ? undefined : { scale: 0.99 }}
          transition={{ duration: 0.25 }}
        >
          Get Started
        </motion.a>
        <motion.a
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="group relative text-sm tracking-wide text-off-white/90"
          whileHover={reducedMotion ? undefined : { opacity: 1 }}
          transition={{ duration: 0.25 }}
        >
          GitHub
          <span className="absolute -bottom-0.5 left-0 h-px w-0 bg-beige/80 transition-all duration-300 group-hover:w-full" />
        </motion.a>
      </motion.div>
    </div>
  )
}
