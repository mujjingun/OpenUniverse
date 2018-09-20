#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 iMVP;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightDir;
    int parallelCount;
    int meridianCount;
    uint noiseIndex;
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2DArray texSamplers[2];

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 worldPos;

const float pi = acos(-1);

mat3 rotationMatrix(vec3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c        );
}

vec2 getTexCoords(vec3 pos) {
    vec3 mapCenterCart = vec3(sin(bounds.mapCenterTheta) * cos(bounds.mapCenterPhi),
                              sin(bounds.mapCenterTheta) * sin(bounds.mapCenterPhi),
                              cos(bounds.mapCenterTheta));
    mat3 rotate = rotationMatrix(normalize(cross(vec3(1, 0, 0), mapCenterCart)), acos(mapCenterCart.x));

    // (theta, phi) in [pi-span, pi+span] x [-span, span] -> [0, 1] x [0, 1]
    vec3 mapCoords = rotate * pos;
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / bounds.mapSpanTheta / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / bounds.mapSpanTheta / 2 + .5f;
    return sphericalCoords;
}

vec2 getOverallTexCoords(vec3 pos) {
    vec3 mapCoords = vec3(pos.zy, -pos.x);
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / pi / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / pi / 2 + .5f;
    return sphericalCoords;
}

float fogFactorExp2(float dist, float density) {
    const float LOG2 = -1.442695;
    float d = density * dist;
    return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
}

int intersect(vec3 p0, vec3 p1, vec3 center, float r, out vec3 point)
{
    vec3 dir = normalize(p1 - p0);
    vec3 diff = center - p0;
    float t0 = dot(diff, dir);
    float d2 = dot(diff, diff) - t0 * t0;
    if (d2 > r * r) {
        return 0;
    }

    float t1 = sqrt(r * r - d2);
    if (t0 < t1) {
        t1 = -t1;
    }
    point = p0 + dir * (t0 - t1);
    return t0 > 0? 1 : -1;
}

const float r = 0.002f;

void main() {
    vec3 cartCoords = normalize(inPos);
    vec2 texCoords = getTexCoords(cartCoords);

    vec4 noiseTex = texture(texSamplers[ubo.noiseIndex], vec3(texCoords, 0));
    float noise = noiseTex.x;

    float h01 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(-1, 0)).x;
    float h21 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(1, 0)).x;
    float h10 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(0, -1)).x;
    float h12 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(0, 1)).x;

    ivec3 size = textureSize(texSamplers[ubo.noiseIndex], 0);
    vec2 grad = vec2((h21 - h01) * float(size.x), (h12 - h10) * float(size.y)) / bounds.mapSpanTheta;
    float biome = noiseTex.w;
    float temp = texture(texSamplers[ubo.noiseIndex], vec3(texCoords, 1)).y;

    vec2 sphere = vec2(acos(cartCoords.z), atan(cartCoords.y, cartCoords.x));
    vec3 dgdt = vec3(cos(sphere.x) * cos(sphere.y), cos(sphere.x) * sin(sphere.y), -sin(sphere.x));
    vec3 dgdp = vec3(sin(sphere.x) * -sin(sphere.y), sin(sphere.x) * cos(sphere.y), 0);
    vec3 dndt = r * grad.x * cartCoords + dgdt * (1 + r * noise);
    vec3 dndp = r * grad.y * cartCoords + dgdp * (1 + r * noise);
    vec3 normal = normalize(cross(dndt, dndp));

    vec3 biomeColor = mix(vec3(61, 82, 48) / 256.0f, vec3(100, 90, 50) / 256.0f, smoothstep(0.3f, 0.35f, biome));

    vec3 sandColor = vec3(236, 221, 166) / 256.0f;
    vec4 terrainColor = vec4(mix(sandColor, biomeColor, smoothstep(0.0f, 0.001f, noise)), 1.0);
    vec4 oceanColor = vec4(vec3(0.2f, 0.2f, 0.8f) * (1.0f - pow(abs(noise), 0.5) * .3f), 1.0f);
    float oceanOrTerrain = noise > 0? 1: 0;
    vec4 color = mix(oceanColor, terrainColor, oceanOrTerrain);
    color = mix(vec4(1.0f), color, smoothstep(0.0f, 0.01f, temp));

    vec3 modelPos = cartCoords * mix(1.0f, 1.0f + noise * r, oceanOrTerrain);
    vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;

    normal = mix(worldPos, normal, oceanOrTerrain);
    float light = max(0.0f, dot(ubo.lightDir.xyz, normal)) + 0.2f;

    vec3 lightReflect = normalize(reflect(ubo.lightDir.xyz, normal));
    vec3 vertexToEye = normalize(worldPos - ubo.eyePos.xyz);
    float specularFactor = dot(vertexToEye, lightReflect);
    specularFactor = pow(max(0, specularFactor), 16);
    light += specularFactor * mix(0.5f, 0.0f, oceanOrTerrain);
    outColor = vec4(light * color.xyz, color.a);

    // fog
    //vec3 exitAtmosphere;
    //int f = intersect(ubo.eyePos.xyz, worldPos, (vec4(0.0f) * ubo.model).xyz, 1.03, exitAtmosphere);
    //float fogDistance = distance(worldPos, ubo.eyePos.xyz);
    //if (f > 0) {
    //    fogDistance -= distance(ubo.eyePos.xyz, exitAtmosphere);
    //}
    //float fogAmount = fogFactorExp2(fogDistance, 10);
    //
    //outColor = mix(outColor, vec4(1.0f), fogAmount);
}

