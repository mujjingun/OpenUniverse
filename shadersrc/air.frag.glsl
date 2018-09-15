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

const float thickness = 0.03f;
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

    vec4 noise = texture(texSamplers[ubo.noiseIndex], texCoords);
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
    float edgeness = D > 0? 0: 1;
    float scatterLength = mix(4.0f * (b - sqrt(max(0, D))), 2.0f * cosine * (1.0f + thickness), edgeness);
    const float maxScatterLength = sqrt(thickness * (2.0f + thickness));
    scatterLength /= maxScatterLength;
    vec4 scatterColor = vec4(0.7f, 0.7f, 1.0f, pow(scatterLength, 3) * 0.1f);
    vec4 cloudColor = mix(vec4(vec3(1.0f), cloud), vec4(0.0f), edgeness);
    outColor.a = scatterColor.a + cloudColor.a * (1.0f - scatterColor.a);
    outColor.rgb = (scatterColor.rgb * scatterColor.a + cloudColor.rgb * cloudColor.a * (1.0f - scatterColor.a)) / outColor.a;
    outColor = vec4(outColor.rgb * light, outColor.a);
}

