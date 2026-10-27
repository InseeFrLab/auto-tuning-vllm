# Auto-Tune vLLM landing page

Static site (Vite + React + Tailwind + Motion) for the project homepage.

## Local development

```bash
npm install
npm run dev
```

Open http://127.0.0.1:5173/

## Production build

```bash
npm run build
npm run preview
```

## GitHub Pages

The site is deployed from `main` by [.github/workflows/website-pages.yml](../.github/workflows/website-pages.yml).

Public URL: **https://inseefrlab.github.io/auto-tuning-vllm/**

In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

After pushing changes under `website/`, check the **Actions** tab for the **Deploy website** workflow.
