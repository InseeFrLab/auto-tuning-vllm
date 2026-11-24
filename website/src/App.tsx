import { Background } from './components/Background'
import { Features } from './components/Features'
import { Hero } from './components/Hero'
import { Nav } from './components/Nav'
import { VideoSection } from './components/VideoSection'

export default function App() {
  return (
    <div className="relative min-h-dvh overflow-x-hidden text-off-white">
      <Background />
      <div className="relative z-10 mx-auto flex min-h-dvh max-w-[90rem] flex-col px-[clamp(1.25rem,4vw,3rem)]">
        <Nav />
        <main className="flex flex-1 flex-col justify-center gap-[clamp(2rem,5vh,3.5rem)] py-[clamp(1.5rem,4vh,3rem)] lg:grid lg:grid-cols-12 lg:content-center lg:items-center lg:gap-8">
          <div className="min-w-0 lg:col-span-5">
            <Hero />
          </div>
          <div className="min-w-0 lg:col-span-7">
            <VideoSection />
          </div>
        </main>
        <Features />
      </div>
    </div>
  )
}
