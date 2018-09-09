#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

const vec3 lightDir = vec3(0.424, 0.566, 0.707);

void main() {
    vec3 dX = dFdx(inPos);
    vec3 dY = dFdy(inPos);
    vec3 normal = normalize(cross(dX,dY));
    //vec3 normal = inNormal;
    float light = max(0.0, dot(lightDir, normal));
    outColor = light * vec4(inColor, 1.0f);
}

