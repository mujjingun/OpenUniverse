#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D bloom;
layout (binding = 1) uniform sampler2D hdr;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

const float exposure = 0.3;

void main()
{
    vec3 hdrColor = texture(hdr, texCoords).rgb;

    vec2 off = vec2(1) / textureSize(bloom, 0);
    vec2 texCoords0 = vec2(texCoords.x / 2, texCoords.y);
    vec2 texCoords1 = vec2(texCoords.x / 4 + .5 + off.x * 5, texCoords.y / 2);
    vec2 texCoords2 = vec2(texCoords.x / 8 + .5 + off.x * 5, texCoords.y / 4 + .5 + off.y * 5);
    vec2 texCoords3 = vec2(texCoords.x / 16 + .5 + off.x * 5, texCoords.y / 8 + .75 + off.y * 10);

    vec3 bloomColor = texture(bloom, texCoords0).rgb * 3
        + texture(bloom, texCoords1).rgb
        + texture(bloom, texCoords2).rgb;

    hdrColor += bloomColor * 2;

    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    outColor = vec4(mapped, 1.0f);
}
