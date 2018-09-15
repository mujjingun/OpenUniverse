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
    uint noiseIndex;
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2D texSamplers[2];

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 inPos;

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

    // (theta, phi) in [0, pi] x [-pi, pi]
    vec3 mapCoords = rotate * pos;
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / bounds.mapSpanTheta / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / bounds.mapSpanTheta / 2 + .5f;
    vec2 texCoords = vec2(sphericalCoords.x, sphericalCoords.y);
    return texCoords;
}

void main() {
    vec3 cartCoords = normalize(inPos);
    vec2 texCoords = getTexCoords(cartCoords);

    vec4 noiseTex = texture(texSamplers[ubo.noiseIndex], texCoords);
    float noise = noiseTex.x;
    float biome = noiseTex.z;
    float temp = noiseTex.w;

    vec3 biomeColor = mix(vec3(61, 82, 48) / 256.0f, vec3(100, 90, 50) / 256.0f, smoothstep(0.3f, 0.35f, biome));

    vec3 sandColor = vec3(236, 221, 166) / 256.0f;
    vec4 terrainColor = vec4(mix(sandColor, biomeColor, smoothstep(0.0f, 0.01f, noise)) - vec3(noise) * .2f, 1.0);
    vec4 oceanColor = vec4(vec3(0.2f, 0.2f, 0.8f) * (1.0f - pow(abs(noise), 0.5) * .3f), 1.0f);
    float oceanOrTerrain = noise > 0? 1: 0;
    vec4 color = mix(oceanColor, terrainColor, oceanOrTerrain);
    color = mix(vec4(1.0f), color, smoothstep(0.0f, 0.01f, temp));

    vec3 modelPos = cartCoords * mix(1.0f, 1.0f + noise * 0.01f, oceanOrTerrain);
    vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;

    vec3 normal = -worldPos; //normalize(cross(dFdx(worldPos), dFdy(worldPos)));
    float light = max(0.0f, dot(ubo.lightDir.xyz, normal)) + 0.1f;

    vec3 lightReflect = normalize(reflect(ubo.lightDir.xyz, normal));
    vec3 vertexToEye = normalize(ubo.eyePos.xyz - worldPos);
    float specularFactor = dot(vertexToEye, lightReflect);
    if (specularFactor > 0) {
        specularFactor = pow(specularFactor, 16);
        light += specularFactor * mix(0.5f, 0.05f, oceanOrTerrain);
    }
    outColor = vec4(light * color.xyz, color.a);
}

