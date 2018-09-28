#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (depth_greater) out float gl_FragDepth;

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
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2DArray texSamplers[2];

layout(set = 1, binding = 0) uniform sampler2D shadowMap;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 inPos;

const float thickness = 0.3f;
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
    if (t0 < t1) {
        t1 = -t1;
    }
    point0 = p0 + dir * (t0 - t1);
    point1 = p0 + dir * (t0 + t1);
    return t0 > 0? 1 : -1;
}

const float lightIntensity = 10.0f;
const float H = 0.01;//0.00131847433; // scale height, units in ER (earth radius)
const vec3 c = vec3(33.10836683, 77.3611417, 188.8702063); // scattering coefficient, units in ER^-1
const int iterations = 10;

float logDensity(float altitude) {
    return -altitude / H;
}

vec3 logTransmittance(vec3 A, vec3 B) {
    float v = 0;
    float dt = 1. / iterations;
    for (float t = 0; t < 1; t += dt) {
        vec3 P = A * t + B * (1 - t);
        v += exp(logDensity(length(P) - 1)) * dt;
    }
    return -c * v * distance(A, B);
}

float gamma(vec3 dir, vec3 L) {
    float cosT = dot(normalize(dir), L);
    float a = 1 + cosT * cosT;
    return 3 / (16 * pi) * (a * a);
}

vec3 scatter(vec3 A, vec3 B) {
    vec3 v = vec3(0);
    float dt = 1. / iterations;

    // light direction in model coordinates
    const vec3 L = normalize((inverse(ubo.model) * -ubo.lightPos).xyz);

    for (float t = 0; t < 1; t += dt) {
        vec3 P = A * t + B * (1 - t);
        vec3 logdF = logTransmittance(P, P - L) + logTransmittance(P, A) + logDensity(length(P) - 1);
        v += exp(logdF) * dt;
    }
    return c * gamma(B - A, L) * v;
}

void main() {
    vec4 unproj = ubo.iMVP * vec4(inPos, 0.0f, 1.0f);
    unproj.xyz /= unproj.w;

    float dist = length(ubo.modelEyePos) - 1;

    vec3 A, B;
    int orientation = intersect(ubo.modelEyePos.xyz, unproj.xyz, vec3(0), 1 + thickness, A, B);
    vec2 texCoords = getOverallTexCoords(normalize(A));
    float cloudNoise = texture(texSamplers[0], vec3(texCoords, 1)).x;

    outColor = vec4(0);

    if (orientation > 0) {
        const vec3 L = normalize((inverse(ubo.model) * -ubo.lightPos).xyz);

        outColor = vec4(scatter(A, B) * lightIntensity * 5, 1);
        //outColor = vec4(vec3(distance(A, B)), 1);
    }

    // the sun
    vec3 worldPos = (ubo.model * unproj).xyz;
    if (intersect(ubo.eyePos.xyz, worldPos, ubo.lightPos.xyz, 10.0, A, B) > 0) {
        outColor += vec4(100.0, 100.0, 100.0, 1.0);
    }
    gl_FragDepth = 1.0f - 0.001f;
}

