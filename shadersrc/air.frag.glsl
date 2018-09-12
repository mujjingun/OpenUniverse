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

const float thickness = 0.03f;
const float pi = 3.1415926536;

void main() {
    vec3 cartCoords = normalize(inPos);

    // (theta, phi) in [0, pi] x [-pi, pi]
    vec2 sphericalCoords = vec2(acos(cartCoords.z), atan(cartCoords.y, cartCoords.x));
    vec2 texCoords = vec2(sphericalCoords.x / pi, sphericalCoords.y / pi / 2 + .5);

    vec4 noise = texture(texSampler, texCoords);
    float cloudNoise = noise.y;
    const float cloud = smoothstep(0.0f, 1.0f, cloudNoise);

    vec3 modelPos = cartCoords;
    vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;
    vec3 normal = normalize(-worldPos);
    float light = max(0.0, dot(ubo.lightDir.xyz, normal));

    vec3 vertexToEye = normalize(ubo.eyePos.xyz - worldPos);
    float cosine = dot(-vertexToEye, normal);
    float b = (1.0f + thickness) * cosine;
    float D = b * b - thickness * (2.0f + thickness);
    float edgeness = smoothstep(0.03, 0.01, D);
    float scatterLength = mix(4.0f * (b - sqrt(max(0, D))), 2.0f * cosine * (1.0f + thickness), edgeness);
    const float maxScatterLength = sqrt(thickness * (2.0f + thickness));
    scatterLength /= maxScatterLength;
    vec4 scatterColor = vec4(0.7f, 0.7f, 1.0f, pow(scatterLength, 3) * 0.1f);
    vec4 cloudColor = mix(vec4(vec3(1.0f), cloud), vec4(0.0f), edgeness);
    outColor.a = scatterColor.a + cloudColor.a * (1.0f - scatterColor.a);
    outColor.rgb = (scatterColor.rgb * scatterColor.a + cloudColor.rgb * cloudColor.a * (1.0f - scatterColor.a)) / outColor.a;
    outColor = vec4(outColor.rgb * light, outColor.a);
}

