#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define WORKGROUP_SIZE 32
layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

layout(binding = 0, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} ubo;

layout(binding = 1, rgba32f) uniform writeonly image2DArray image;

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
{
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

float ridgeNoise(vec3 v)
{
    return 2 * (.5 - abs(0.5 - snoise(v)));
}

float ridgeWithOctaves(vec3 v, int n)
{
    float F = 1;
    float coeff = 1.0f;
    for (int i = 0; i < n; ++i) {
        float t = ridgeNoise(v * coeff) / coeff;
        t = sign(t) * pow(abs(t), 0.8f);
        F += t * F;
        coeff *= 2;
    }
    F = sign(F) * pow(abs(F), 1.3f);
    return F;
}

// make rotation matrix
mat3 rotationMatrix(vec3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c        );
}

const float pi = acos(-1);

void main()
{
    ivec3 size = imageSize(image);

    // [0, 1] x [0, 1]
    vec2 inPos = vec2(float(gl_GlobalInvocationID.x) / size.x, float(gl_GlobalInvocationID.y) / size.y);

    vec3 mapCenterCart = vec3(sin(ubo.mapCenterTheta) * cos(ubo.mapCenterPhi),
                              sin(ubo.mapCenterTheta) * sin(ubo.mapCenterPhi),
                              cos(ubo.mapCenterTheta));
    mat3 rotate = rotationMatrix(normalize(cross(vec3(1, 0, 0), mapCenterCart)), -acos(mapCenterCart.x));

    const vec2 spherical = vec2(
                mix(pi / 2 - ubo.mapSpanTheta, pi / 2 + ubo.mapSpanTheta, inPos.x), // [pi/2 - span, pi/2 + span]
                mix(-ubo.mapSpanTheta, ubo.mapSpanTheta, inPos.y)); // [-span, span]
    const vec3 cartesian = rotate * vec3(sin(spherical.x) * cos(spherical.y), sin(spherical.x) * sin(spherical.y), cos(spherical.x));

    // terrain
    vec3 seed_1 = cartesian * 2 + vec3(-4.0f);
    float noise_1 = ridgeWithOctaves(seed_1, 12) - max(0, snoise(seed_1 / 2.0f)) * 3.0f - 0.2f;

    // cloud
    const vec3 seed_2 = seed_1 * 2 + vec3(10.0f);
    const float noise_2 = smoothstep(-0.5, .3, snoise(seed_2)) *
            (snoise(seed_2 * 2) / 2
            + snoise(seed_2 * 4) / 4
            + snoise(seed_2 * 8) / 8
            + snoise(seed_2 * 32) / 16);

    // biome
    const vec3 seed_3 = cartesian + vec3(-5.0f);
    const float noise_3 = snoise(seed_3)
            + snoise(seed_3 * 2) / 2
            + snoise(seed_3 * 8) / 4;

    // temperature
    const vec3 seed_4 = cartesian * 2 + vec3(-5.0f);
    const float noise_4 = sqrt(1 - cartesian.z * cartesian.z) * 30.0f - 10.0f - max(noise_1, 0) * 3.0f
            + snoise(seed_4) * 2.0f
            + snoise(seed_4 * 2) * 1.0f;

    imageStore(image, ivec3(gl_GlobalInvocationID.xy, 0), vec4(noise_1, 1.0f, 1.0f, noise_3));
    imageStore(image, ivec3(gl_GlobalInvocationID.xy, 1), vec4(noise_2, noise_4, 1.0f, 1.0f));
}
