#version 440
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
} ubo;

// number of control points in the output patch
layout (vertices = 4) out;

layout(location = 0) in vec3 inPos[];

layout (location = 0) out vec3 outPos[];

in gl_PerVertex
{
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];

float GetTessLevel(float dist)
{
    if (dist < 0.1f) {
        return 5.0f;
    }
    if (dist < 0.3f) {
        return 4.0f;
    }
    if (dist < 1.0f) {
        return 3.0f;
    }
    else if (dist < 4.0f) {
        return 2.0f;
    }
    return 1.0f;
}

// The TCS gets each triangle as a patch with 3 CPs and passes it through to the TES.
void main(void)
{
    outPos[gl_InvocationID] = inPos[gl_InvocationID];

    const vec3 centerPos = (inPos[0] + inPos[1] + inPos[2] + inPos[3]) / 4.0f;
    const vec3 worldPos = (ubo.model * vec4(centerPos, 1.0f)).xyz;
    const float avgDist = distance(worldPos, ubo.eyePos.xyz);
    const float tessLevel = GetTessLevel(avgDist);

    gl_TessLevelOuter[0] = tessLevel;
    gl_TessLevelOuter[1] = tessLevel;
    gl_TessLevelOuter[2] = tessLevel;
    gl_TessLevelOuter[3] = tessLevel;
    gl_TessLevelInner[0] = tessLevel;
    gl_TessLevelInner[1] = tessLevel;
}
