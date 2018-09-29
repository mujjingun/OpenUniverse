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

const float thickness = 0.01f;
const float pi = acos(-1);

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
    //if (t0 < t1) t1 = -t1;
    point0 = p0 + dir * (t0 - t1);
    point1 = p0 + dir * (t0 + t1);
    return t0 > 0? 1 : -1;
}

const float lightIntensity = 10.0f;
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

float gamma(vec3 dir, vec3 L) {
    float cosT = dot(normalize(dir), L);
    float a = 1 + cosT * cosT;
    return 3 / (16 * pi) * a;
}

vec3 scatter(vec3 A, vec3 B, vec3 L) {
    vec3 v = vec3(0);
    float dt = 1. / 10;

    // numerical integration
    for (float t = 0; t < 1; t += dt) {
        vec3 P = B * (t + dt/2) + A * (1 - t - dt/2);
        vec3 logdF = vec3(0);
        logdF += logTransmittance(A, P);
        logdF += logDensity(length(P) - 1) + 1;

        vec3 _, C;
        intersect(P, P - L, vec3(0), 1 + thickness, _, C);
        //if (intersect(P, P - L, vec3(0), 1, _, _) != 0) {
            logdF += logTransmittance(P, C);
            v += exp(logdF) * dt;
        //}
    }
    return c * gamma(B - A, L) * v * distance(A, B);
}

void main() {
    vec4 unproj = ubo.iMVP * vec4(inPos, 1.0f, 1.0f);
    unproj.xyz /= unproj.w;

    float dist = length(ubo.modelEyePos) - 1;

    vec3 A0, B0, A1, B1, A, B;
    vec3 viewRay = normalize(unproj.xyz - ubo.modelEyePos.xyz);
    int outer = intersect(ubo.modelEyePos.xyz, unproj.xyz, vec3(0), 1 + thickness, A0, B0);
    int inner = intersect(ubo.modelEyePos.xyz, unproj.xyz, vec3(0), 1, A1, B1);

    if (dist > thickness && outer > 0) {
        if (inner == 0) {
            A = A0;
            B = B0;
        }
        else if (inner > 0) {
            A = A0;
            B = A1;
        }
    } else if (dist <= thickness) {
        if (inner > 0) {
            A = ubo.modelEyePos.xyz;
            B = A1;
        }
        else {
            A = ubo.modelEyePos.xyz;
            B = B0;
        }
    }

    vec2 texCoords = getOverallTexCoords(normalize(A));
    float cloudNoise = texture(texSamplers[0], vec3(texCoords, 1)).x;

    outColor = vec4(0, 0, 0, 1);

    if (dist > thickness && outer > 0 || dist <= thickness) {
        // light direction in model coordinates
        const vec3 L = normalize(-(inverse(ubo.model) * ubo.lightPos).xyz);

        vec3 atmosphere = scatter(A, B, L) * lightIntensity;
        vec3 _, C;
        intersect(B, B - L, vec3(0), 1 + thickness, _, C);
        float tr = dot(exp(logTransmittance(A, B) + logTransmittance(B, C)), vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(atmosphere, 1);
    }

    // the sun
    vec3 worldPos = (ubo.model * unproj).xyz;
    if (intersect(ubo.eyePos.xyz, worldPos, ubo.lightPos.xyz, 10.0, A, B) > 0) {
        outColor += vec4(100.0, 100.0, 100.0, 1.0);
    }
    gl_FragDepth = 0.0f;
}

