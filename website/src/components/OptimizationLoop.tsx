import { motion, useReducedMotion } from 'motion/react'

const LOOP_PATH_ID = 'optimization-loop-path'

/** Closed loop: Optuna → vLLM → GuideLLM → Results → Optuna */
const LOOP_PATH =
  'M 72 118 C 120 88, 200 88, 248 118 C 278 138, 278 182, 248 202 C 200 232, 120 232, 72 202 C 42 182, 42 138, 72 118 Z'

const NODES = [
  { label: 'Optuna', x: 72, y: 118, align: 'start' as const },
  { label: 'vLLM', x: 248, y: 118, align: 'end' as const },
  { label: 'GuideLLM', x: 248, y: 202, align: 'end' as const },
  { label: 'Results', x: 72, y: 202, align: 'start' as const },
]

export function OptimizationLoop() {
  const reducedMotion = useReducedMotion()

  return (
    <motion.div
      className="relative mx-auto flex w-full max-w-md items-center justify-center lg:mx-0 lg:max-w-none lg:justify-end lg:pr-[clamp(0rem,3vw,3rem)]"
      initial={reducedMotion ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.75,
        delay: 0.2,
        ease: [0.22, 1, 0.36, 1],
      }}
      aria-hidden
    >
      <svg
        viewBox="0 0 320 320"
        className="h-[clamp(14rem,32vh,22rem)] w-[clamp(14rem,32vh,22rem)] max-w-full"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <path id={LOOP_PATH_ID} d={LOOP_PATH} />
        </defs>

        <motion.path
          d={LOOP_PATH}
          stroke="rgba(203, 187, 166, 0.22)"
          strokeWidth="1"
          strokeLinecap="round"
          initial={reducedMotion ? false : { pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.2, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
        />

        {!reducedMotion && (
          <>
            <circle r="2.5" fill="rgba(244, 239, 230, 0.85)">
              <animateMotion
                dur="28s"
                repeatCount="indefinite"
                calcMode="linear"
              >
                <mpath href={`#${LOOP_PATH_ID}`} />
              </animateMotion>
            </circle>
            <circle r="2" fill="rgba(203, 187, 166, 0.7)">
              <animateMotion
                dur="28s"
                repeatCount="indefinite"
                calcMode="linear"
                begin="-14s"
              >
                <mpath href={`#${LOOP_PATH_ID}`} />
              </animateMotion>
            </circle>
          </>
        )}

        {NODES.map((node, index) => (
          <g key={node.label}>
            <motion.circle
              cx={node.x}
              cy={node.y}
              r="5"
              fill="#12100e"
              stroke="rgba(203, 187, 166, 0.45)"
              strokeWidth="1"
              initial={reducedMotion ? false : { scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{
                duration: 0.45,
                delay: 0.4 + index * 0.08,
                ease: [0.22, 1, 0.36, 1],
              }}
              style={{ transformOrigin: `${node.x}px ${node.y}px` }}
            />
            <motion.text
              x={node.x + (node.align === 'start' ? -10 : 10)}
              y={node.y + (node.label === 'Optuna' || node.label === 'vLLM' ? -14 : 22)}
              textAnchor={node.align === 'start' ? 'end' : 'start'}
              className="fill-off-white/90 text-[11px] tracking-wide"
              style={{ fontFamily: "'IBM Plex Sans', system-ui, sans-serif" }}
              initial={reducedMotion ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.5 + index * 0.08 }}
            >
              {node.label}
            </motion.text>
          </g>
        ))}

        {!reducedMotion &&
          NODES.map((node, index) => (
            <motion.circle
              key={`pulse-${node.label}`}
              cx={node.x}
              cy={node.y}
              r="5"
              fill="none"
              stroke="rgba(203, 187, 166, 0.25)"
              strokeWidth="1"
              animate={{ r: [5, 9], opacity: [0.35, 0] }}
              transition={{
                duration: 3.5,
                repeat: Infinity,
                delay: index * 0.85,
                ease: 'easeOut',
              }}
            />
          ))}
      </svg>
    </motion.div>
  )
}
