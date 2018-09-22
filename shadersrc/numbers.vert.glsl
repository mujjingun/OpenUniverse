#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

const vec2 coords[] = vec2[](
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),
    vec2(1, -1)
);

void main() {
    gl_Position = vec4(coords[gl_VertexIndex], 0.0f, 1.0f);
}
