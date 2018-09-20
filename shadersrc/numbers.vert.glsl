#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 3, std140) uniform Digits {
    uint n;
} digits[3];

const vec2 coords[] = vec2[](
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),
    vec2(1, -1)
);

const vec2 coords2[] = vec2[](
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),
    vec2(1, -1)
);

void main() {
    gl_Position = vec4(coords2[gl_VertexIndex], 0.0f, 1.0f);
}
