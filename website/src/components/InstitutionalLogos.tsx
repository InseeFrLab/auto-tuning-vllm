import { motion, useReducedMotion } from 'motion/react'

const logoBase =
  'w-auto transition-opacity duration-300 contrast-[1.05] saturate-[0.92]'

export function InstitutionalLogos() {
  const reducedMotion = useReducedMotion()
  const base = import.meta.env.BASE_URL

  return (
    <motion.div
      className="pointer-events-auto flex items-center justify-end gap-[clamp(1.25rem,3vw,2.25rem)] pt-4 lg:absolute lg:right-0 lg:bottom-[clamp(1.25rem,3vh,2rem)] lg:pt-0"
      initial={reducedMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <a
        href="https://www.insee.fr/fr/accueil"
        target="_blank"
        rel="noopener noreferrer"
        className="leading-none"
        aria-label="INSEE (opens in a new tab)"
      >
        <img
          src={`${base}logos/insee.svg`}
          alt=""
          className={`${logoBase} h-[clamp(1.5rem,3vw,2.25rem)] opacity-[0.85] brightness-[2.45] contrast-[1.12] saturate-[1.08] hover:opacity-95`}
          width={120}
          height={40}
          loading="lazy"
          decoding="async"
        />
      </a>
      <a
        href="https://www.economie.gouv.fr/dgfip"
        target="_blank"
        rel="noopener noreferrer"
        className="leading-none"
        aria-label="Direction générale des Finances publiques (opens in a new tab)"
      >
        <img
          src={`${base}logos/dgfip.png`}
          alt=""
          className={`${logoBase} h-[clamp(1.625rem,3.2vw,2.375rem)] opacity-[0.72] brightness-[2.15] saturate-[0.85] hover:opacity-90`}
          width={306}
          height={163}
          loading="lazy"
          decoding="async"
        />
      </a>
    </motion.div>
  )
}
