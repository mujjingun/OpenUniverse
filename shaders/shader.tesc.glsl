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

// The TCS gets each triangle as a patch with 3 CPs and passes it through to the TES.
void main(void)
{
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outColor[gl_InvocationID] = inColor[gl_InvocationID];

    float eyeDist0 = distance(ubo.eyePos, outPos[0]);
    float eyeDist1 = distance(ubo.eyePos, outPos[1]);
    float eyeDist2 = distance(ubo.eyePos, outPos[2]);

    gl_TessLevelOuter[0] = 100;
    gl_TessLevelOuter[1] = 100;
    gl_TessLevelOuter[2] = 100;
    gl_TessLevelInner[0] = 100;
}
