import { dirname } from "path";
import { fileURLToPath } from "url";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static export: `next build` emits ./out, deployed straight to
  // Cloudflare Pages. The viewer only fetches precomputed files, so there is
  // no server, no inference backend, and nothing to pay for.
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  images: { unoptimized: true },
  // Pin the workspace root to this folder so a stray parent lockfile does not
  // confuse Next's root inference.
  outputFileTracingRoot: dirname(fileURLToPath(import.meta.url)),
};

export default nextConfig;
