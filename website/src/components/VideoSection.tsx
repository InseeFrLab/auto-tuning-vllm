import { useRef, useState } from 'react'
import videoUrl from '../assets/auto-tune-vllm.mp4?url'
import posterUrl from '../assets/auto-tune-vllm-poster.jpg?url'

export function VideoSection() {
  const player = useRef<HTMLVideoElement>(null)
  const [showControls, setShowControls] = useState(false)

  function startVideo() {
    const video = player.current
    if (!video) return

    setShowControls(true)
    video.focus()
    void video.play().catch(() => {
      // Native controls remain available if the browser blocks this play attempt.
    })
  }

  return (
    <section id="video" aria-labelledby="video-title" className="w-full">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2
          id="video-title"
          className="font-display text-[clamp(1.25rem,6.4vw,1.5rem)] tracking-tight text-off-white sm:text-3xl"
        >
          How Auto-Tune vLLM works
        </h2>
        <span className="font-mono-label text-xs tracking-wider text-beige">2 min</span>
      </div>

      <div className="relative overflow-hidden rounded-lg border border-beige/25 bg-warm-black">
        <video
          ref={player}
          aria-labelledby="video-title"
          className="block aspect-video w-full"
          controls={showControls}
          playsInline
          tabIndex={showControls ? 0 : -1}
          preload="none"
          poster={posterUrl}
          src={videoUrl}
        >
          <a href={videoUrl}>Download the Auto-Tune vLLM video</a>
        </video>

        {!showControls && (
          <button
            type="button"
            aria-label="Play the Auto-Tune vLLM video"
            onClick={startVideo}
            className="group absolute inset-0 grid cursor-pointer place-items-center bg-warm-black focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-beige"
          >
            <img src={posterUrl} alt="" className="absolute inset-0 size-full object-cover" />
            <span
              aria-hidden="true"
              className="relative grid size-13 place-items-center rounded-full border border-beige/40 bg-off-white shadow-lg shadow-black/40 transition-colors group-hover:bg-beige sm:size-15"
            >
              <span className="ml-0.5 size-0 border-y-[0.45rem] border-y-transparent border-l-[0.7rem] border-l-warm-black" />
            </span>
          </button>
        )}
      </div>
    </section>
  )
}
