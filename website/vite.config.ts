import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// GITHUB_PAGES=true in CI sets base for https://inseefrlab.github.io/auto-tuning-vllm/
export default defineConfig({
  base: process.env.GITHUB_PAGES ? '/auto-tuning-vllm/' : '/',
  plugins: [react(), tailwindcss()],
})
