#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D bloom[];
layout (binding = 1) uniform sampler2D hdr;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

const float exposure = 0.3;
const float gamma = 2.2;

const float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

#define getBloom(index) \
    ( weight[0] * texture(bloom[index], texCoords).rgb + \
    + weight[1] * textureOffset(bloom[index], texCoords, ivec2(0, +1)).rgb \
    + weight[1] * textureOffset(bloom[index], texCoords, ivec2(0, -1)).rgb \
    + weight[2] * textureOffset(bloom[index], texCoords, ivec2(0, +2)).rgb \
    + weight[2] * textureOffset(bloom[index], texCoords, ivec2(0, -2)).rgb \
    + weight[3] * textureOffset(bloom[index], texCoords, ivec2(0, +3)).rgb \
    + weight[3] * textureOffset(bloom[index], texCoords, ivec2(0, -3)).rgb \
    + weight[4] * textureOffset(bloom[index], texCoords, ivec2(0, +4)).rgb \
    + weight[4] * textureOffset(bloom[index], texCoords, ivec2(0, -4)).rgb)

void main()
{
    vec3 hdrColor = texture(hdr, texCoords).rgb;
    hdrColor += getBloom(0) * 2 + getBloom(1) + getBloom(2);

    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    //mapped = pow(mapped, vec3(1.0 / gamma));
    outColor = vec4(mapped, 1.0f);
}
