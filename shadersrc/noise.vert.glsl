#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} ubo;

layout(location = 0) out vec2 outPos;

out gl_PerVertex
{
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
};

const vec2 positions[4] = vec2[](
    vec2(0, 0),
    vec2(0, 1),
    vec2(1, 1),
    vec2(1, 0)
);

void main() {
    outPos = positions[gl_VertexIndex];
    gl_Position = vec4(outPos * 2 - vec2(1), 0.0f, 1.0f);
}
