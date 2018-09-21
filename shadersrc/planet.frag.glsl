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

vec2 getTexCoords(vec3 mapCoords) {
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

const float r = 0.002f;

// returns unnormalized normal
vec3 getNormal(vec3 coords, float noise, vec2 grad) {
    vec2 sphere = vec2(acos(coords.z), atan(coords.y, coords.x));
    vec3 dgdt = vec3(cos(sphere.x) * cos(sphere.y), cos(sphere.x) * sin(sphere.y), -sin(sphere.x));
    vec3 dgdp = vec3(sin(sphere.x) * -sin(sphere.y), sin(sphere.x) * cos(sphere.y), 0);
    vec3 dndt = r * grad.x * coords + dgdt * (1 + r * noise);
    vec3 dndp = r * grad.y * coords + dgdp * (1 + r * noise);
    return cross(dndt, dndp);
}

vec2 getGradient(vec2 texCoords) {
    const float h01 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(-1, 0)).x;
    const float h21 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(1, 0)).x;
    const float h10 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(0, -1)).x;
    const float h12 = textureOffset(texSamplers[ubo.noiseIndex], vec3(texCoords, 0), ivec2(0, 1)).x;

    const ivec3 size = textureSize(texSamplers[ubo.noiseIndex], 0);
    return vec2(h21 - h01, h12 - h10) * size.xy / bounds.mapSpanTheta;
}

void main() {
    vec3 cartCoords = normalize(inPos);

    vec3 mapCenterCart = vec3(sin(bounds.mapCenterTheta) * cos(bounds.mapCenterPhi),
                              sin(bounds.mapCenterTheta) * sin(bounds.mapCenterPhi),
                              cos(bounds.mapCenterTheta));

    const vec3 axis = normalize(vec3(0, -mapCenterCart.z, mapCenterCart.y));
    const float s = sqrt(1 - mapCenterCart.x * mapCenterCart.x);
    const float c = mapCenterCart.x;
    const float oc = 1.0 - c;

    const mat3 rotate = mat3(c, -axis.z * s, axis.y * s,
                             axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z,
                             -axis.y * s, oc * axis.y * axis.z, oc * axis.z * axis.z + c);
    const mat3 irotate = mat3(c, axis.z * s, -axis.y * s,
                              -axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z,
                              axis.y * s, oc * axis.y * axis.z, oc * axis.z * axis.z + c);

    vec3 mapCoords = rotate * cartCoords;
    vec2 texCoords = getTexCoords(mapCoords);
    vec4 noiseTex = texture(texSamplers[ubo.noiseIndex], vec3(texCoords, 0));

    float noise = noiseTex.x;
    vec2 grad = getGradient(texCoords);

    vec3 normal = normalize(mat3(ubo.model) * irotate * getNormal(mapCoords, noise, grad));
    vec3 flatNormal = normalize(mat3(ubo.model) * cartCoords);

    // biome
    float biome = noiseTex.w;
    vec3 biomeColor = mix(vec3(61, 82, 48) / 256.0f, vec3(100, 90, 50) / 256.0f, smoothstep(0.3f, 0.35f, biome));

    // ocean
    vec3 sandColor = vec3(236, 221, 166) / 256.0f;
    vec4 terrainColor = vec4(mix(sandColor, biomeColor, smoothstep(0.0f, 0.001f, noise)), 1.0);
    vec4 oceanColor = vec4(vec3(0.2f, 0.2f, 0.8f) * (1.0f - pow(abs(noise), 0.5) * .3f), 1.0f);
    float oceanOrTerrain = noise > 0? 1: 0;
    vec4 color = mix(oceanColor, terrainColor, oceanOrTerrain);

    // ice
    float temp = texture(texSamplers[ubo.noiseIndex], vec3(texCoords, 1)).y;
    color = mix(vec4(1.0f), color, smoothstep(0.0f, 0.01f, temp));

    // lighting
    vec3 modelPos = cartCoords * mix(1.0f, 1.0f + noise * r, oceanOrTerrain);
    vec3 worldPos = mat3(ubo.model) * modelPos;

    normal = mix(flatNormal, normal, oceanOrTerrain);
    float light = max(0.0f, dot(ubo.lightDir.xyz, normal)) + 0.05f;

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

