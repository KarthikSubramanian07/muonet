// Colormap lookup tables, built as 256-wide RGBA textures for the raymarch
// shader. Stops are sampled from matplotlib's perceptually-uniform maps plus a
// custom "cosmic" ramp that leans into the muon theme. Denser = hotter reads
// as the correct intuition for scattering density.

export type ColormapName = "inferno" | "viridis" | "cosmic" | "gray";

type Stop = [number, [number, number, number]];

const STOPS: Record<ColormapName, Stop[]> = {
  inferno: [
    [0.0, [0, 0, 4]],
    [0.25, [40, 11, 84]],
    [0.5, [101, 21, 110]],
    [0.65, [159, 42, 99]],
    [0.8, [212, 72, 66]],
    [0.9, [245, 125, 21]],
    [1.0, [252, 255, 164]],
  ],
  viridis: [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ],
  cosmic: [
    [0.0, [4, 4, 16]],
    [0.35, [26, 22, 92]],
    [0.6, [86, 51, 200]],
    [0.8, [190, 74, 220]],
    [0.92, [255, 120, 170]],
    [1.0, [255, 236, 200]],
  ],
  gray: [
    [0.0, [10, 10, 14]],
    [1.0, [245, 245, 250]],
  ],
};

export function buildColormapLUT(name: ColormapName, size = 256): Uint8Array {
  const stops = STOPS[name];
  const lut = new Uint8Array(size * 4);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let j = 0;
    while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
    const [t0, c0] = stops[j];
    const [t1, c1] = stops[j + 1];
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    lut[i * 4 + 0] = Math.round(c0[0] + (c1[0] - c0[0]) * f);
    lut[i * 4 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * f);
    lut[i * 4 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * f);
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

export const COLORMAP_NAMES: ColormapName[] = ["inferno", "cosmic", "viridis", "gray"];
