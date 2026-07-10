"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { buildColormapLUT, type ColormapName } from "@/lib/colormaps";
import { fragmentShader, vertexShader } from "@/lib/shaders";

const RENDER_MODES = { dvr: 0, mip: 1, iso: 2 } as const;
export type RenderMode = keyof typeof RENDER_MODES;

interface Props {
  texture: THREE.Data3DTexture | null;
  colormap: ColormapName;
  threshold: number;
  opacity: number;
  mode: RenderMode;
  autoRotate: boolean;
}

export default function VolumeCanvas(props: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  // The render loop is created once, so read live-changing values through refs.
  const autoRotateRef = useRef(props.autoRotate);
  autoRotateRef.current = props.autoRotate;
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    material: THREE.ShaderMaterial;
    cmapTexture: THREE.DataTexture;
    raf: number;
  } | null>(null);

  // One-time setup.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    if (!document.createElement("canvas").getContext("webgl2")) {
      mount.innerHTML =
        '<div style="padding:24px;color:var(--fg-muted);font-size:14px">This viewer needs WebGL2, which your browser or GPU does not support.</div>';
      return;
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(1.6, 1.1, 1.8);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.9;
    controls.maxDistance = 6;

    const cmapData = buildColormapLUT(props.colormap);
    const cmapTexture = new THREE.DataTexture(cmapData, 256, 1, THREE.RGBAFormat);
    cmapTexture.needsUpdate = true;

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      transparent: true,
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        u_data: { value: null },
        u_cmap: { value: cmapTexture },
        u_clim: { value: new THREE.Vector2(0.0, 1.0) },
        u_threshold: { value: props.threshold },
        u_opacity: { value: props.opacity },
        u_steps: { value: 192 },
        u_mode: { value: RENDER_MODES[props.mode] },
      },
      vertexShader,
      fragmentShader,
    });

    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    scene.add(mesh);

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const animate = () => {
      controls.autoRotate = autoRotateRef.current;
      controls.autoRotateSpeed = 1.4;
      controls.update();
      renderer.render(scene, camera);
      const s = stateRef.current;
      if (s) s.raf = requestAnimationFrame(animate);
    };

    stateRef.current = { renderer, scene, camera, controls, material, cmapTexture, raf: 0 };
    stateRef.current.raf = requestAnimationFrame(animate);

    return () => {
      observer.disconnect();
      const s = stateRef.current;
      if (s) cancelAnimationFrame(s.raf);
      controls.dispose();
      material.dispose();
      mesh.geometry.dispose();
      cmapTexture.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to prop changes without rebuilding the scene.
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.material.uniforms.u_data.value = props.texture;
    s.material.uniforms.u_threshold.value = props.threshold;
    s.material.uniforms.u_opacity.value = props.opacity;
    s.material.uniforms.u_mode.value = RENDER_MODES[props.mode];
  }, [props.texture, props.threshold, props.opacity, props.mode]);

  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    const data = buildColormapLUT(props.colormap);
    (s.cmapTexture.image.data as Uint8Array).set(data);
    s.cmapTexture.needsUpdate = true;
  }, [props.colormap]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%" }} />;
}
