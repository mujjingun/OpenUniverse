#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (quads, fractional_even_spacing, ccw) in;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 eyePos;
} ubo;

layout (location = 0) in vec3 inPos[];

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 seed;
layout (location = 2) out vec3 vertexToEye;

out gl_PerVertex {
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
};

vec3 interpolate3D(vec3 v0, vec3 v1, vec3 v2, vec3 v3)
{
    const vec3 p0 = mix(v0, v3, gl_TessCoord.x);
    const vec3 p1 = mix(v1, v2, gl_TessCoord.x);
    return mix(p0, p1, gl_TessCoord.y);
}

// The PG subdivided an equilateral triangle into
// smaller triangles and executes the TES for every generated vertex.
void main(void)
{
    outPos = normalize(interpolate3D(inPos[0], inPos[1], inPos[2], inPos[3]));
    seed = outPos * 4 + vec3(10.0f);

    const float thickness = 0.03f;
    const vec3 worldPos = (ubo.model * vec4(outPos * (1.0f + thickness), 1.0f)).xyz;
    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0f);
    vertexToEye = normalize(ubo.eyePos - worldPos);
}
