#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 eyePos;
} ubo;

// number of control points in the output patch
layout (vertices = 3) out;

layout(location = 0) in vec3 inPos[];
layout(location = 1) in vec3 inColor[];

layout (location = 0) out vec3 outPos[];
layout (location = 1) out vec3 outColor[];

in gl_PerVertex
{
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];

float GetTessLevel(float Distance0, float Distance1)
{
    float AvgDistance = (Distance0 + Distance1) / 2.0f;
    return 10.0f / AvgDistance;
}

// The TCS gets each triangle as a patch with 3 CPs and passes it through to the TES.
void main(void)
{
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outColor[gl_InvocationID] = inColor[gl_InvocationID];

    const vec3 worldPos0 = (ubo.model * vec4(inPos[0], 1.0f)).xyz;
    const vec3 worldPos1 = (ubo.model * vec4(inPos[1], 1.0f)).xyz;
    const vec3 worldPos2 = (ubo.model * vec4(inPos[2], 1.0f)).xyz;
    float eyeDist0 = distance(ubo.eyePos, worldPos0);
    float eyeDist1 = distance(ubo.eyePos, worldPos1);
    float eyeDist2 = distance(ubo.eyePos, worldPos2);

    gl_TessLevelOuter[0] = GetTessLevel(eyeDist1, eyeDist2);
    gl_TessLevelOuter[1] = GetTessLevel(eyeDist2, eyeDist0);
    gl_TessLevelOuter[2] = GetTessLevel(eyeDist0, eyeDist1);
    gl_TessLevelInner[0] = gl_TessLevelOuter[2] * 1.5f;
}
