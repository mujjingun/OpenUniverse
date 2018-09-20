#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (depth_greater) out float gl_FragDepth;

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

layout(location = 0) in vec2 inPos;

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

vec2 getOverallTexCoords(vec3 pos) {
    vec3 mapCoords = vec3(pos.zy, -pos.x);
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / pi / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / pi / 2 + .5f;
    return sphericalCoords;
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

void main() {
    vec4 unproj0 = ubo.iMVP * vec4(inPos, 0.0f, 1.0f);
    unproj0.xyz /= unproj0.w;

    vec3 cartCoords;
    float dist = length(ubo.modelEyePos) - 1;

    // inside the atmosphere
    if (dist < thickness) {
        int orientation = intersect(ubo.modelEyePos.xyz, unproj0.xyz, vec3(0), 1 + thickness, cartCoords);
        if (orientation == 0) {
            discard;
        }

        vec2 texCoords = getOverallTexCoords(normalize(cartCoords));

        vec4 noise = texture(texSamplers[0], vec3(texCoords, 1));
        float cloudNoise = noise.x;

        const float cloud = smoothstep(0.0f, 1.0f, cloudNoise) * 0.9f;

        const vec3 modelPos = cartCoords;
        const vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;
        const vec3 normal = normalize(worldPos);
        const float light = max(0.0, dot(ubo.lightDir.xyz, normal));

        vec4 scatterColor = vec4(0.3f, 0.3f, 1.0f, 1.0f);
        vec4 cloudColor = vec4(1.0f, 1.0f, 1.0f, cloud);
        outColor.a = cloudColor.a + scatterColor.a * (1.0f - cloudColor.a);
        outColor.rgb = (cloudColor.rgb * cloudColor.a + scatterColor.rgb * scatterColor.a * (1.0f - cloudColor.a)) / outColor.a;
        outColor = vec4(outColor.rgb * light, outColor.a);
        gl_FragDepth = 1.0f - 0.001f;
    }
    // outside the atmosphere
    if (dist >= thickness) {
        int orientation = intersect(ubo.modelEyePos.xyz, unproj0.xyz, vec3(0), 1 + thickness, cartCoords);
        if (orientation != 1) {
            discard;
        }

        vec2 texCoords = getOverallTexCoords(normalize(cartCoords));

        vec4 noise = texture(texSamplers[0], vec3(texCoords, 1));
        float cloudNoise = noise.x;

        const float cloud = smoothstep(0.0f, 1.0f, cloudNoise) * 0.9f;

        const vec3 modelPos = cartCoords;
        const vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;
        const vec3 normal = normalize(worldPos);
        const float light = max(0.0, dot(ubo.lightDir.xyz, normal));

        vec3 vertexToEye = normalize(ubo.eyePos.xyz - worldPos);
        float cosine = dot(vertexToEye, normal);
        float b = (1.0f + thickness) * cosine;
        float D = b * b - thickness * (2.0f + thickness);
        float edgeness = D > 0? 0: 1;
        float scatterLength = mix(2.0f * (b - sqrt(max(0, D))), 2.0f * cosine * (1.0f + thickness), edgeness);
        const float maxScatterLength = sqrt(thickness * (2.0f + thickness));
        scatterLength /= maxScatterLength;
        vec4 scatterColor = vec4(0.7f, 0.7f, 1.0f, pow(scatterLength, 3) * 0.1f);
        vec4 cloudColor = mix(vec4(vec3(1.0f), cloud), vec4(0.0f), edgeness);
        outColor.a = scatterColor.a + cloudColor.a * (1.0f - scatterColor.a);
        outColor.rgb = (scatterColor.rgb * scatterColor.a + cloudColor.rgb * cloudColor.a * (1.0f - scatterColor.a)) / outColor.a;
        outColor = vec4(outColor.rgb * light, outColor.a);
        gl_FragDepth = 0;
    }
}

