// A single-pass raymarching volume renderer for a scalar Data3DTexture.
//
// The mesh is a unit cube centered at the origin; the camera orbits it, so the
// cube's local frame equals world space and the ray origin is just
// cameraPosition. Three render modes share one march loop:
//   0 DVR  front-to-back alpha compositing (the soft "cloud that resolves")
//   1 MIP  maximum intensity projection
//   2 ISO  first-crossing isosurface with gradient shading
//
// GLSL ES 3.00 (set material.glslVersion = THREE.GLSL3).

export const vertexShader = /* glsl */ `
out vec3 vPos;
void main() {
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vPos;
out vec4 fragColor;

uniform sampler3D u_data;
uniform sampler2D u_cmap;
uniform vec2 u_clim;
uniform float u_threshold;
uniform float u_opacity;
uniform int u_steps;
uniform int u_mode;

vec2 boxIntersect(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 a = (vec3(-0.5) - ro) * inv;
  vec3 b = (vec3(0.5) - ro) * inv;
  vec3 tmin = min(a, b);
  vec3 tmax = max(a, b);
  float tn = max(max(tmin.x, tmin.y), tmin.z);
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  return vec2(tn, tf);
}

float sampleScalar(vec3 uvw) {
  float raw = texture(u_data, uvw).r;
  return clamp((raw - u_clim.x) / max(u_clim.y - u_clim.x, 1e-4), 0.0, 1.0);
}

vec3 gradient(vec3 uvw, float h) {
  float dx = sampleScalar(uvw + vec3(h, 0.0, 0.0)) - sampleScalar(uvw - vec3(h, 0.0, 0.0));
  float dy = sampleScalar(uvw + vec3(0.0, h, 0.0)) - sampleScalar(uvw - vec3(0.0, h, 0.0));
  float dz = sampleScalar(uvw + vec3(0.0, 0.0, h)) - sampleScalar(uvw - vec3(0.0, 0.0, h));
  return vec3(dx, dy, dz);
}

void main() {
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vPos - ro);
  vec2 t = boxIntersect(ro, rd);
  float tnear = max(t.x, 0.0);
  float tfar = t.y;
  if (tfar <= tnear) discard;

  int steps = u_steps;
  float dt = (tfar - tnear) / float(steps);
  vec3 p = ro + rd * tnear;
  vec3 dstep = rd * dt;

  vec4 acc = vec4(0.0);
  float mip = 0.0;

  for (int i = 0; i < 512; i++) {
    if (i >= steps) break;
    vec3 uvw = p + 0.5;
    float v = sampleScalar(uvw);

    if (u_mode == 1) {
      mip = max(mip, v);
    } else if (v > u_threshold) {
      vec3 c = texture(u_cmap, vec2(v, 0.5)).rgb;
      if (u_mode == 2) {
        vec3 n = normalize(-gradient(uvw, 1.0 / 32.0) + 1e-6);
        vec3 lightDir = normalize(vec3(0.6, 0.8, 0.5));
        float diff = clamp(dot(n, lightDir), 0.0, 1.0);
        fragColor = vec4(c * (0.35 + 0.75 * diff), 1.0);
        return;
      }
      float a = v * v * u_opacity;
      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.98) break;
    }
    p += dstep;
  }

  if (u_mode == 1) {
    if (mip <= u_threshold) discard;
    fragColor = vec4(texture(u_cmap, vec2(mip, 0.5)).rgb, 1.0);
  } else {
    if (acc.a < 0.01) discard;
    fragColor = vec4(acc.rgb / max(acc.a, 1e-4), acc.a);
  }
}
`;
