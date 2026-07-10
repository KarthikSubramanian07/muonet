# Muonet viewer

An interactive 3D viewer for Muonet reconstructions. It is a fully static
Next.js site (`output: export`) with a Three.js raymarching volume renderer. No
server, no inference backend: it fetches precomputed voxel volumes as static
files, which is what keeps it free to host on Cloudflare Pages.

## Develop

```bash
npm install
npm run dev        # http://localhost:3000
```

The viewer reads volumes from `public/volumes/`. Generate them from the Python
side:

```bash
# from the repo root
muonet export --checkpoint checkpoints/muonet.pt --out viewer/public/volumes
```

## Build and deploy

```bash
npm run build      # emits ./out (static)
npm run deploy     # builds, then wrangler pages deploy out --project-name muonet
```

`npm run deploy` needs a Cloudflare login once (`npx wrangler login`) or a
`CLOUDFLARE_API_TOKEN` in the environment. The GitHub Action in
`.github/workflows/deploy-viewer.yml` does the same automatically on push to
`main` once the two repository secrets are set.

## How the renderer works

- Volumes load into a `THREE.Data3DTexture` (8-bit, one byte per voxel, x
  fastest so it maps straight into the texture).
- A single fragment shader raymarches a unit cube (`lib/shaders.ts`) in three
  modes: DVR alpha compositing, MIP, and gradient-shaded isosurface.
- Colormaps are built as 256-wide lookup textures (`lib/colormaps.ts`).
- The canvas is a client-only component loaded with `dynamic(..., { ssr: false })`
  so WebGL is never touched during the static prerender.
