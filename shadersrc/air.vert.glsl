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

layout(location = 0) out vec3 unproj;
layout(location = 1) out vec3 L;
layout(location = 2) out vec2 outPos;

void main() {
    outPos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0f + -1.0f;

    vec4 unproj0 = ubo.iMVP * vec4(outPos, 1.0f, 1.0f);
    unproj = unproj0.xyz / unproj0.w;

    // light direction in model coordinates
    L = normalize(-(inverse(ubo.model) * ubo.lightPos).xyz);

    gl_Position = vec4(outPos, 0.0f, 1.0f);
}
