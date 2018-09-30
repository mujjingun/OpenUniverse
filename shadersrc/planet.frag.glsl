#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 iMVP;
    mat4 shadowVP;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightPos;
    int parallelCount;
    int meridianCount;
    uint noiseIndex;
    float terrainFactor;
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2DArray texSamplers[2];

layout(set = 1, binding = 0) uniform sampler2DShadow shadowMap;

layout(location = 0) out vec4 outColor;
//layout(depth_any) out float gl_FragDepth;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in float logz;

const float pi = acos(-1);

vec2 getTexCoords(vec3 mapCoords, float span) {
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / span / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / span / 2 + .5f;
    return sphericalCoords;
}

void getMapCoords(vec3 pos, float centerT, float centerP, float span, out vec3 mapCoords, out vec2 texCoords, out mat3 irotate)
{
    const vec3 center = vec3(sin(centerT) * cos(centerP), sin(centerT) * sin(centerP), cos(centerT));

    const vec3 axis = normalize(vec3(0, -center.z, center.y));
    const float s = sqrt(1 - center.x * center.x);
    const float c = center.x;
    const float oc = 1.0 - c;

    const mat3 rotate = mat3(c, -axis.z * s, axis.y * s,
                             axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z,
                             -axis.y * s, oc * axis.y * axis.z, oc * axis.z * axis.z + c);
    irotate = mat3(c, axis.z * s, -axis.y * s,
                   -axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z,
                   axis.y * s, oc * axis.y * axis.z, oc * axis.z * axis.z + c);

    mapCoords = rotate * pos;
    texCoords = getTexCoords(mapCoords, span);
}

const float lightIntensity = 10.0f;

// returns unnormalized normal
vec3 getNormal(vec3 coords, float noise, vec2 grad) {
    const vec2 sphere = vec2(acos(coords.z), atan(coords.y, coords.x));
    const vec3 dgdt = vec3(cos(sphere.x) * cos(sphere.y), cos(sphere.x) * sin(sphere.y), -sin(sphere.x));
    const vec3 dgdp = vec3(sin(sphere.x) * -sin(sphere.y), sin(sphere.x) * cos(sphere.y), 0);
    const vec3 dndt = ubo.terrainFactor * grad.x * coords + dgdt * (1 + ubo.terrainFactor * noise);
    const vec3 dndp = ubo.terrainFactor * grad.y * coords + dgdp * (1 + ubo.terrainFactor * noise);
    return cross(dndt, dndp);
}

vec2 getGradient(uint idx, vec2 texCoords, float span) {
    const float h01 = textureOffset(texSamplers[idx], vec3(texCoords, 0), ivec2(-1, 0)).x;
    const float h21 = textureOffset(texSamplers[idx], vec3(texCoords, 0), ivec2(1, 0)).x;
    const float h10 = textureOffset(texSamplers[idx], vec3(texCoords, 0), ivec2(0, -1)).x;
    const float h12 = textureOffset(texSamplers[idx], vec3(texCoords, 0), ivec2(0, 1)).x;

    const ivec3 size = textureSize(texSamplers[idx], 0);
    return vec2(h21 - h01, h12 - h10) * size.xy / span;
}

const float H = 0.00131847433; // scale height, units in ER (earth radius)
const vec3 c = vec3(33.10836683, 77.3611417, 188.8702063); // scattering coefficient, units in ER^-1

float logDensity(float altitude) {
    return -max(altitude, 0) / H;
}

vec3 logTransmittance(vec3 A, vec3 B) {
    float v = 0;
    float dt = 1. / 10;
    for (float t = 0; t < 1; t += dt) {
        vec3 P = B * (t + dt/2) + A * (1 - t - dt/2);
        v += exp(logDensity(length(P) - 1)) * dt;
    }
    return -c * v * distance(A, B);
}

int intersect(vec3 p0, vec3 p1, vec3 center, float r, out vec3 point0, out vec3 point1)
{
    vec3 dir = normalize(p1 - p0);
    vec3 diff = center - p0;
    float t0 = dot(diff, dir);
    float d2 = dot(diff, diff) - t0 * t0;
    if (d2 > r * r) {
        return 0;
    }

    float t1 = sqrt(r * r - d2);
    //if (t0 < t1) t1 = -t1;
    point0 = p0 + dir * (t0 - t1);
    point1 = p0 + dir * (t0 + t1);
    return t0 > 0? 1 : -1;
}

void main() {
    vec3 cartCoords = normalize(inPos);

    float span = bounds.mapSpanTheta;
    uint texIndex = ubo.noiseIndex;
    vec3 mapCoords;
    vec2 texCoords;
    mat3 irotate;

    getMapCoords(cartCoords, bounds.mapCenterTheta, bounds.mapCenterPhi, bounds.mapSpanTheta,
        mapCoords, texCoords, irotate);

    if (texCoords.x < 0 || texCoords.x > 1 || texCoords.y < 0 || texCoords.y > 1) {
        getMapCoords(cartCoords, 0, 0, pi, mapCoords, texCoords, irotate);
        texIndex = 0;
        span = pi;
    }

    vec4 noiseTex = texture(texSamplers[texIndex], vec3(texCoords, 0));
    vec4 noiseTex1 = texture(texSamplers[texIndex], vec3(texCoords, 1));
    float terrain = noiseTex.x;

    vec2 grad = getGradient(texIndex, texCoords, span);
    vec3 normal = normalize(mat3(ubo.model) * irotate * getNormal(mapCoords, terrain, grad));
    vec3 flatNormal = normalize(mat3(ubo.model) * cartCoords);

    float oceanOrTerrain = terrain > 0? 1: 0;
    normal = mix(flatNormal, normal, oceanOrTerrain);

    // biome
    float biome = noiseTex.w;
    vec3 biomeColor = mix(vec3(41, 62, 28) / 256.0f, vec3(100, 90, 50) / 256.0f, smoothstep(0.3f, 0.35f, biome));

    // ocean
    vec3 sandColor = vec3(236, 221, 166) / 256.0f;
    vec4 terrainColor = vec4(mix(sandColor, biomeColor, smoothstep(0.0f, 0.001f, terrain)), 1.0);
    vec4 oceanColor = vec4(vec3(0.2f, 0.2f, 0.8f) * (1.0f - pow(abs(terrain), 0.5) * .3f), 1.0f);
    vec4 color = mix(oceanColor, terrainColor, oceanOrTerrain);

    // ice
    float temp = noiseTex1.y;
    color = mix(vec4(1.0f), color, smoothstep(0.0f, 0.01f, temp));

    // lighting
    vec3 modelPos = cartCoords * mix(1.0f, 1.0f + terrain * ubo.terrainFactor, oceanOrTerrain);
    vec3 worldPos = mat3(ubo.model) * modelPos;

    vec3 lightDir = normalize(ubo.lightPos.xyz - worldPos);
    float cosLightAngle = dot(normal, lightDir);
    float light = max(0, cosLightAngle) * lightIntensity;

    vec3 lightReflect = normalize(reflect(lightDir, normal));
    vec3 vertexToEye = normalize(worldPos - ubo.eyePos.xyz);
    float specularFactor = dot(vertexToEye, lightReflect);
    specularFactor = pow(max(0, specularFactor), 32);
    light += specularFactor * mix(lightIntensity, 0.0f, oceanOrTerrain);

    // shadowing
    vec4 shadowPos = ubo.shadowVP * vec4(worldPos, 1);
    shadowPos.xyz /= shadowPos.w;
    float bias = clamp(0.005 * tan(acos(clamp(-cosLightAngle, 0, 1))), 0, 0.02);
    float visibility = texture(shadowMap, vec3(shadowPos.xy / 2 + .5, shadowPos.z - bias));

    light = max(0.001, light * visibility);

    // scattering
    const vec3 L = normalize(modelPos - (inverse(ubo.model) * ubo.lightPos).xyz);
    vec3 _, C;
    intersect(modelPos, modelPos - L, vec3(0), 1 + 0.01, _, C);
    vec3 vLight = light * exp(logTransmittance(modelPos, C));

    outColor = vec4(vLight * color.rgb, color.a);
    //outColor = vec4(mod(modelPos, vec3(0.1)) * 10, 1.0);
    gl_FragDepth = logz;
}
