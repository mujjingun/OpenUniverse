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
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2DArray texSamplers[2];

layout(set = 1, binding = 0) uniform sampler2D shadowMap;

layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInput depthBuffer;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 unproj;
layout(location = 1) in vec3 L;
layout(location = 2) in vec2 outPos;

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
const vec3 c = vec3(33, 77, 188); // scattering coefficient, units in ER^-1

float density(float altitude) {
    return exp(-max(altitude, 0) / H);
}

vec3 logTransmittance(vec3 A, vec3 B) {
    float v = 0;
    float dt = 1. / 10;
    for (float t = 0; t < 1; t += dt) {
        vec3 P = B * (t + dt/2) + A * (1 - t - dt/2);
        v += density(length(P) - 1) * dt;
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

        vec3 _, C;
        intersect(P, P - L, vec3(0), 1 + thickness, _, C);
        //if (intersect(P, P - L, vec3(0), 1, _, _) != 0) {
        logdF += logTransmittance(P, C);
        v += exp(logdF) * density(length(P) - 1) * dt;
        //}
    }
    return c * gamma(B - A, L) * v * distance(A, B);
}

const float C = 1;
const float far = 10000.0;

void main() {
    float dist = length(ubo.modelEyePos) - 1;

    vec3 A0, B0, A1, B1, A, B;
    vec3 viewRay = normalize(unproj.xyz - ubo.modelEyePos.xyz);
    int outer = intersect(ubo.modelEyePos.xyz, unproj.xyz, vec3(0), 1 + thickness, A0, B0);
    int inner = intersect(ubo.modelEyePos.xyz, unproj.xyz, vec3(0), 1, A1, B1);

    // reconstruct model position from depth buffer
    float ldepth = subpassLoad(depthBuffer).r;
    const float FC = 1.0 / log(far * C + 1);
    float w = (exp(ldepth / FC) - 1) / C;
    float z = 10 / (10 - .1) + 10 * .1 / (.1 - 10) / w;
    vec4 pos = vec4(outPos, z, 1.0);
    vec4 modelCoords = ubo.iMVP * pos;
    vec3 modelPos = modelCoords.xyz / modelCoords.w;

    if (dist > thickness && outer > 0) {
        if (ldepth == 1.0) {
            A = A0;
            B = B0;
        }
        else {
            A = A0;
            B = modelPos;
        }
    } else if (dist <= thickness) {
        if (ldepth < 1.0) {
            A = ubo.modelEyePos.xyz;
            B = modelPos;
        }
        else {
            A = ubo.modelEyePos.xyz;
            B = B0;
        }
    }

    vec2 texCoords = getOverallTexCoords(normalize(A));
    float cloudNoise = texture(texSamplers[0], vec3(texCoords, 1)).x;

    bool render = false;
    outColor = vec4(0, 0, 0, 1);

    if (dist > thickness && outer > 0 || dist <= thickness) {
        vec3 atmosphere = scatter(A, B, L) * lightIntensity * 5;
        vec3 _, C;
        intersect(B, B - L, vec3(0), 1 + thickness, _, C);
        float tr = dot(exp(logTransmittance(A, B) + logTransmittance(B, C)), vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(atmosphere, 1);
        render = true;
    }

    // the sun
    vec3 worldPos = mat3(ubo.model) * unproj;
    if (intersect(ubo.eyePos.xyz, worldPos, ubo.lightPos.xyz, 10.0, A, B) > 0) {
        outColor += vec4(100.0, 100.0, 100.0, 0.0);
        render = true;
    }

    if (!render) {
        discard;
    }
}

