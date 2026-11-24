import { motion, useReducedMotion } from 'motion/react'

const logoPlate =
  'flex h-9 shrink-0 items-center rounded-sm bg-off-white px-2 transition-opacity duration-300 hover:opacity-85'

export function InstitutionalLogos() {
  const reducedMotion = useReducedMotion()
  const base = import.meta.env.BASE_URL

  return (
    <motion.div
      className="pointer-events-auto flex shrink-0 items-center justify-end gap-[clamp(0.75rem,2vw,1.5rem)] pt-4 xl:pt-0"
      initial={reducedMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <a
        href="https://www.insee.fr/fr/accueil"
        target="_blank"
        rel="noopener noreferrer"
        className={logoPlate}
        aria-label="INSEE (opens in a new tab)"
      >
        <img
          src={`${base}logos/insee.svg`}
          alt=""
          className="h-6 w-auto"
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
        className={logoPlate}
        aria-label="Direction générale des Finances publiques (opens in a new tab)"
      >
        <img
          src={`${base}logos/dgfip.png`}
          alt=""
          className="h-7 w-auto"
          width={306}
          height={163}
          loading="lazy"
          decoding="async"
        />
      </a>
      <img
        src={`${base}logos/quickscale-ai.png`}
        alt="Quickscale AI"
        className="w-[clamp(4.75rem,8vw,6rem)] opacity-85"
        width={1000}
        height={144}
        loading="lazy"
        decoding="async"
      />
    </motion.div>
  )
}
