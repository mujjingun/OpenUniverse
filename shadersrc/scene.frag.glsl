#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D bloom;
layout (binding = 1) uniform sampler2D hdr;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

const float exposure = 0.3;

const float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

#define getBloom(coords, xoff, yoff) \
    ( weight[0] * textureOffset(bloom, coords, ivec2(xoff, yoff)).rgb + \
    + weight[1] * textureOffset(bloom, coords, ivec2(xoff, yoff+1)).rgb \
    + weight[1] * textureOffset(bloom, coords, ivec2(xoff, yoff-1)).rgb \
    + weight[2] * textureOffset(bloom, coords, ivec2(xoff, yoff+2)).rgb \
    + weight[2] * textureOffset(bloom, coords, ivec2(xoff, yoff-2)).rgb \
    + weight[3] * textureOffset(bloom, coords, ivec2(xoff, yoff+3)).rgb \
    + weight[3] * textureOffset(bloom, coords, ivec2(xoff, yoff-3)).rgb \
    + weight[4] * textureOffset(bloom, coords, ivec2(xoff, yoff+4)).rgb \
    + weight[4] * textureOffset(bloom, coords, ivec2(xoff, yoff-4)).rgb)

void main()
{
    vec3 hdrColor = texture(hdr, texCoords).rgb;

    vec2 texCoords0 = vec2(texCoords.x / 2, texCoords.y);
    vec2 texCoords1 = vec2(texCoords.x / 4 + .5, texCoords.y / 2);
    vec2 texCoords2 = vec2(texCoords.x / 8 + .5, texCoords.y / 4 + .5);

    vec3 bloomColor = getBloom(texCoords0,0,0) * 2 + getBloom(texCoords1,3,0) + getBloom(texCoords2,3,3);

    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    //mapped = pow(mapped, vec3(1.0 / gamma));
    outColor = vec4(mapped + bloomColor, 1.0f);
}
