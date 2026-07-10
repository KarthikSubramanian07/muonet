"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type * as THREE from "three";

import { COLORMAP_NAMES, type ColormapName } from "@/lib/colormaps";
import {
  loadManifest,
  loadVolume,
  preloadVolumes,
  type Manifest,
  type MethodVolume,
} from "@/lib/volume";
import type { RenderMode } from "./VolumeCanvas";

// Three.js touches the DOM/WebGL, so the canvas is client-only and never
// prerendered during static export.
const VolumeCanvas = dynamic(() => import("./VolumeCanvas"), { ssr: false });

const BASE = "/volumes";
const METHOD_LABELS: Record<string, string> = {
  ground_truth: "Ground truth",
  poca: "PoCA (classical)",
  mlem: "MLEM (classical)",
  learned: "Learned (ours)",
};

function useTexture(stem: string | null): THREE.Data3DTexture | null {
  const [tex, setTex] = useState<THREE.Data3DTexture | null>(null);
  useEffect(() => {
    let alive = true;
    if (!stem) return;
    loadVolume(BASE, stem)
      .then((v) => alive && setTex(v.texture))
      .catch(() => alive && setTex(null));
    return () => {
      alive = false;
    };
  }, [stem]);
  return tex;
}

export default function Viewer() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sceneIdx, setSceneIdx] = useState(0);
  const [dosageIdx, setDosageIdx] = useState(0);
  const [method, setMethod] = useState("learned");
  const [colormap, setColormap] = useState<ColormapName>("inferno");
  const [mode, setMode] = useState<RenderMode>("dvr");
  const [threshold, setThreshold] = useState(0.12);
  const [opacity, setOpacity] = useState(0.9);
  const [autoRotate, setAutoRotate] = useState(true);
  const [sideBySide, setSideBySide] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  useEffect(() => {
    const stored = (localStorage.getItem("muonet-theme") as "light" | "dark") || null;
    const initial =
      stored ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setTheme(initial);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("muonet-theme", theme);
  }, [theme]);

  useEffect(() => {
    loadManifest(BASE)
      .then(setManifest)
      .catch(() => setError("Reconstruction volumes are not built yet. Run: muonet export"));
  }, []);

  const scene = manifest?.scenes[sceneIdx] ?? null;
  const dosage = manifest?.dosages[dosageIdx] ?? 0;
  const methods = manifest?.methods ?? ["poca"];
  const effectiveMethod = methods.includes(method) ? method : methods[0];

  // Preload every dosage for the current scene/method so the slider is instant.
  useEffect(() => {
    if (!scene || !manifest) return;
    const stems: string[] = [scene.ground_truth];
    for (const d of manifest.dosages) {
      const m = scene.volumes[String(d)]?.[effectiveMethod];
      if (m) stems.push(m.stem);
    }
    preloadVolumes(BASE, stems);
  }, [scene, manifest, effectiveMethod]);

  const current: MethodVolume | undefined =
    scene?.volumes[String(dosage)]?.[effectiveMethod];
  const pocaCurrent = scene?.volumes[String(dosage)]?.poca;
  const learnedCurrent = scene?.volumes[String(dosage)]?.learned;

  const mainTex = useTexture(
    effectiveMethod === "ground_truth" ? (scene?.ground_truth ?? null) : (current?.stem ?? null),
  );
  const pocaTex = useTexture(pocaCurrent?.stem ?? null);
  const learnedTex = useTexture(learnedCurrent?.stem ?? null);

  const canvasProps = { colormap, threshold, opacity, mode, autoRotate };

  const hasLearned = methods.includes("learned");

  const metrics = useMemo(() => {
    if (!current) return null;
    return { psnr: current.psnr, ssim: current.ssim, iou: current.iou, accuracy: current.accuracy };
  }, [current]);

  return (
    <div className="shell">
      <div className="topbar">
        <div className="brand">
          <span className="dot" /> muonet
        </div>
        <nav>
          <a href="https://github.com/KarthikSubramanian07/muonet" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <button
            className="icon-btn"
            aria-label="Toggle theme"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </nav>
      </div>

      <header className="hero">
        <div className="eyebrow">muon scattering tomography</div>
        <h1>See inside anything dense.</h1>
        <p className="lede">
          Cosmic muons rain through everything on Earth and bend, ever so slightly, in dense
          matter. Muonet turns those bends into a 3D picture. Drag the dosage slider and watch a
          blur resolve into an object as more muons arrive.
        </p>
      </header>

      {error ? (
        <div className="note">{error}</div>
      ) : (
        <div className="stage">
          {sideBySide ? (
            <div className="dual">
              <Panel label="PoCA" metric={pocaCurrent}>
                <VolumeCanvas texture={pocaTex} {...canvasProps} />
              </Panel>
              <Panel label={hasLearned ? "Learned" : "PoCA"} metric={learnedCurrent ?? pocaCurrent}>
                <VolumeCanvas texture={learnedTex ?? pocaTex} {...canvasProps} />
              </Panel>
            </div>
          ) : (
            <div className="canvas-card">
              <div className="canvas-badge mono">
                {scene?.label ?? "…"} · {METHOD_LABELS[effectiveMethod] ?? effectiveMethod} ·{" "}
                {dosage} muons
              </div>
              <VolumeCanvas texture={mainTex} {...canvasProps} />
            </div>
          )}

          <div className="panel">
            <h2>Controls</h2>

            <div className="field">
              <label>Scene</label>
              <select
                value={sceneIdx}
                onChange={(e) => setSceneIdx(Number(e.target.value))}
              >
                {manifest?.scenes.map((s, i) => (
                  <option key={s.name} value={i}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>
                Dosage <span className="value mono">{dosage} muons</span>
              </label>
              <input
                type="range"
                min={0}
                max={(manifest?.dosages.length ?? 1) - 1}
                step={1}
                value={dosageIdx}
                onChange={(e) => setDosageIdx(Number(e.target.value))}
              />
            </div>

            {!sideBySide && (
              <div className="field">
                <label>Method</label>
                <div className="segment small">
                  {["ground_truth", ...methods].map((m) => (
                    <button
                      key={m}
                      aria-pressed={effectiveMethod === m}
                      onClick={() => setMethod(m)}
                    >
                      {m === "ground_truth" ? "Truth" : m === "learned" ? "Learned" : m.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="toggle-row">
              <span>Compare PoCA vs Learned</span>
              <button
                className="switch"
                aria-pressed={sideBySide}
                aria-label="Toggle side by side"
                onClick={() => setSideBySide(!sideBySide)}
                disabled={!hasLearned}
              />
            </div>

            <div className="field">
              <label>Render</label>
              <div className="segment small">
                {(["dvr", "mip", "iso"] as RenderMode[]).map((m) => (
                  <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <label>Colormap</label>
              <select value={colormap} onChange={(e) => setColormap(e.target.value as ColormapName)}>
                {COLORMAP_NAMES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>
                Density threshold <span className="value mono">{threshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min={0}
                max={0.8}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </div>

            <div className="toggle-row">
              <span>Auto-rotate</span>
              <button
                className="switch"
                aria-pressed={autoRotate}
                aria-label="Toggle auto-rotate"
                onClick={() => setAutoRotate(!autoRotate)}
              />
            </div>

            {metrics && (
              <div className="metrics">
                <Metric k="PSNR" v={metrics.psnr != null ? `${metrics.psnr.toFixed(1)} dB` : "—"} />
                <Metric k="SSIM" v={metrics.ssim?.toFixed(3) ?? "—"} />
                <Metric k="IoU" v={metrics.iou?.toFixed(3) ?? "—"} />
                <Metric k="high-Z acc" v={metrics.accuracy?.toFixed(3) ?? "—"} />
              </div>
            )}
          </div>
        </div>
      )}

      <div className="note">
        <strong>What you are looking at.</strong> These are real reconstructions from Muonet&apos;s
        analytic simulator, not renders of the ground truth. The reconstructed quantity is scattering
        density (inverse radiation length), which is what muon scattering physically measures. Every
        volume was precomputed offline and is served as a static file, so this page costs nothing to
        run. The learned model is a trajectory MLP plus a compact 3D U-Net, an architecture from
        prior work; the contribution here is the reproducible, detector-realistic package around it.
      </div>

      <footer>
        <span>Muonet · MIT licensed · built by Karthik Subramanian</span>
        <span className="mono">cosmic rays in, 3D interiors out</span>
      </footer>
    </div>
  );
}

function Panel({
  label,
  metric,
  children,
}: {
  label: string;
  metric?: MethodVolume;
  children: React.ReactNode;
}) {
  return (
    <div className="canvas-card">
      <div className="canvas-badge mono">
        {label}
        {metric?.psnr != null ? ` · ${metric.psnr.toFixed(1)} dB` : ""}
      </div>
      {children}
    </div>
  );
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div className="metric">
      <div className="k">{k}</div>
      <div className="v mono">{v}</div>
    </div>
  );
}
