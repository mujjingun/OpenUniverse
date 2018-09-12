#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (quads, fractional_even_spacing, ccw) in;

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightDir;
    int parallelCount;
    int meridianCount;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec3 inPos[];

layout (location = 0) out vec3 outPos;

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

const float pi = 3.1415926536;

// The PG subdivided an equilateral triangle into
// smaller triangles and executes the TES for every generated vertex.
void main(void)
{
    const vec3 pos = normalize(interpolate3D(inPos[0], inPos[1], inPos[2], inPos[3]));

    // (theta, phi) in [0, pi] x [-pi, pi]
    vec2 sphericalCoords = vec2(acos(pos.z), atan(pos.y, pos.x));
    vec2 texCoords = vec2(sphericalCoords.x / pi, sphericalCoords.y / pi / 2 + .5);

    const float noise = max(0, texture(texSampler, texCoords).x);

    vec3 modelPos = pos * (1.0f + vec3(noise * 0.015f));
    vec3 worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;
    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0f);
    outPos = pos;
}
