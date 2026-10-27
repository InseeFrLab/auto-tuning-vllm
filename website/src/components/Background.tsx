import {
  motion,
  useMotionTemplate,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from 'motion/react'
import { useEffect } from 'react'

export function Background() {
  const reducedMotion = useReducedMotion()
  const pointerX = useMotionValue(0.5)
  const pointerY = useMotionValue(0.5)

  const springX = useSpring(pointerX, { stiffness: 40, damping: 22, mass: 0.8 })
  const springY = useSpring(pointerY, { stiffness: 40, damping: 22, mass: 0.8 })

  const glowLeft = useTransform(springX, (v) => `${v * 100}%`)
  const glowTop = useTransform(springY, (v) => `${v * 100}%`)
  const parallaxX = useTransform(springX, (v) => `${(v - 0.5) * 24}px`)
  const parallaxY = useTransform(springY, (v) => `${(v - 0.5) * 16}px`)

  const glowBackground = useMotionTemplate`radial-gradient(600px circle at ${glowLeft} ${glowTop}, rgba(203, 187, 166, 0.09), transparent 65%)`

  useEffect(() => {
    if (reducedMotion) return
    const onPointerMove = (event: PointerEvent) => {
      pointerX.set(event.clientX / window.innerWidth)
      pointerY.set(event.clientY / window.innerHeight)
    }
    window.addEventListener('pointermove', onPointerMove, { passive: true })
    return () => window.removeEventListener('pointermove', onPointerMove)
  }, [pointerX, pointerY, reducedMotion])

  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden>
      <div className="absolute inset-0 bg-warm-black" />
      <div
        className="absolute inset-0 opacity-40"
        style={{
          background:
            'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(203, 187, 166, 0.06), transparent 70%)',
        }}
      />
      <motion.div
        className="absolute inset-[-10%]"
        style={
          reducedMotion
            ? undefined
            : { x: parallaxX, y: parallaxY }
        }
      >
        <div
          className="h-full w-full opacity-70"
          style={{
            background:
              'radial-gradient(ellipse 50% 45% at 70% 35%, rgba(42, 51, 46, 0.55), transparent 70%)',
          }}
        />
      </motion.div>
      {!reducedMotion && (
        <motion.div
          className="absolute inset-0"
          style={{ background: glowBackground }}
        />
      )}
    </div>
  )
}
