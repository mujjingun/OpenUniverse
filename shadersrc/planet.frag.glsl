#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightDir;
    int parallelCount;
    int meridianCount;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 seed;
layout(location = 2) in vec3 vertexToEye;

const float pi = 3.1415926536;

void main() {
    vec3 cartCoords = normalize(inPos);

    // (theta, phi) in [0, pi] x [-pi, pi]
    vec2 sphericalCoords = vec2(acos(cartCoords.z), atan(cartCoords.y, cartCoords.x));
    vec2 texCoords = vec2(sphericalCoords.x / pi, sphericalCoords.y / pi / 2 + .5);

    vec4 noiseTex = texture(texSampler, texCoords);
    float noise = noiseTex.x;
    float biome = noiseTex.z;

    vec3 biomeColor = mix(vec3(61, 82, 48) / 256.0f, vec3(100, 90, 50) / 256.0f, smoothstep(0.3f, 0.35f, biome));

    vec3 sandColor = vec3(236, 221, 166) / 256.0f;
    vec4 terrainColor = vec4(mix(sandColor, biomeColor, smoothstep(0.0f, 0.07f, noise)), 1.0);
    vec4 oceanColor = vec4(vec3(0.2f, 0.2f, 0.85f) * (1.0f - pow(abs(noise), 0.5) * .3f), 0.9f - 0.2 * smoothstep(-0.2, 0, noise));
    float oceanOrTerrain = smoothstep(-0.01f, 0.01f, noise);
    vec4 color = mix(oceanColor, terrainColor, oceanOrTerrain);

    vec3 modelPos = cartCoords * mix(1.0f, 1.0f + noise * 0.015f, oceanOrTerrain);
    vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;

    vec3 normal = normalize(cross(dFdx(worldPos), dFdy(worldPos)));
    float light = max(0.0f, dot(ubo.lightDir.xyz, normal)) + 0.1f;

    vec3 lightReflect = normalize(reflect(ubo.lightDir.xyz, normal));
    float specularFactor = dot(vertexToEye, lightReflect);
    if (specularFactor > 0) {
        specularFactor = pow(specularFactor, 16);
        light += specularFactor * mix(0.5f, 0.05f, oceanOrTerrain);
    }
    outColor = vec4(light * color.xyz, color.a);
}

