#include "terrain.h"

#define GLM_FORCE_SWIZZLE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

using namespace glm;

static vec3 mod289(vec3 x)
{
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

static vec4 mod289(vec4 x)
{
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

static vec4 permute(vec4 x)
{
    return mod289(((x * 34.0f) + 1.0f) * x);
}

static vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159f - 0.85373472095314f * r;
}

static float snoise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i = floor(v + dot(v, C.yyy()));
    vec3 x0 = v - i + dot(i, C.xxx());

    // Other corners
    vec3 g = step(x0.yzx(), x0.xyz());
    vec3 l = 1.0f - g;
    vec3 i1 = min(g.xyz(), l.zxy());
    vec3 i2 = max(g.xyz(), l.zxy());

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    // Permutations
    i = mod289(i);
    vec4 p = permute(permute(permute(
                                 i.z + vec4(0.0, i1.z, i2.z, 1.0))
                         + i.y + vec4(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857f; // 1.0/7.0
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0f * floor(p * ns.z * ns.z); //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0f * x_); // mod(j,N)

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0f - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0f + 1.0f;
    vec4 s1 = floor(b1) * 2.0f + 1.0f;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6f - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0f);
    m = m * m;
    return 42.0f * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

static float ridgeNoise(vec3 v)
{
    return 2.0f * (.5f - abs(0.5f - snoise(v)));
}

static float ridgeWithOctaves(vec3 v, int n)
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

float ou::terrainHeight(float theta, float phi, float factor)
{
    vec3 cartesian = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    vec3 seed = cartesian * 2.0f + vec3(-4.0f);
    float noise = ridgeWithOctaves(seed, 20) - max(0.0f, snoise(seed / 2.0f)) * 3.0f - 0.2f;
    return noise * factor;
}
