// Load an exported Muonet volume (a .bin of uint8 voxels plus a .json header)
// into a Three.js Data3DTexture. The binary is laid out x-fastest, which is
// exactly what Data3DTexture(data, width, height, depth) expects.

import * as THREE from "three";

export interface VolumeHeader {
  shape: [number, number, number];
  dtype: string;
  layout: string;
  value_range: [number, number];
  bin: string;
}

export interface LoadedVolume {
  texture: THREE.Data3DTexture;
  header: VolumeHeader;
}

export interface MethodVolume {
  stem: string;
  psnr?: number;
  ssim?: number;
  iou?: number;
  accuracy?: number;
}

export interface ManifestEntry {
  name: string;
  label: string;
  ground_truth: string;
  // volumes[dosage][method] -> volume + metrics
  volumes: Record<string, Record<string, MethodVolume>>;
}

export interface Manifest {
  scenes: ManifestEntry[];
  dosages: number[];
  methods: string[];
  detector: string;
}

const cache = new Map<string, LoadedVolume>();

export async function loadManifest(baseUrl: string): Promise<Manifest> {
  const res = await fetch(`${baseUrl}/manifest.json`, { cache: "force-cache" });
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  return res.json();
}

export async function loadVolume(baseUrl: string, stem: string): Promise<LoadedVolume> {
  const key = `${baseUrl}/${stem}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const header: VolumeHeader = await fetch(`${key}.json`, { cache: "force-cache" }).then((r) => {
    if (!r.ok) throw new Error(`header fetch failed: ${r.status}`);
    return r.json();
  });
  const buffer = await fetch(`${baseUrl}/${header.bin}`, { cache: "force-cache" }).then((r) => {
    if (!r.ok) throw new Error(`volume fetch failed: ${r.status}`);
    return r.arrayBuffer();
  });

  const [nx, ny, nz] = header.shape;
  const data = new Uint8Array(buffer);
  const texture = new THREE.Data3DTexture(data, nx, ny, nz);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;

  const loaded = { texture, header };
  cache.set(key, loaded);
  return loaded;
}

export function preloadVolumes(baseUrl: string, stems: string[]): void {
  for (const stem of stems) void loadVolume(baseUrl, stem).catch(() => {});
}
