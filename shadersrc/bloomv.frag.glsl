#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D bloom;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

const float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main()
{
    vec3 bloomColor = weight[0] * texture(bloom, texCoords).rgb;
    bloomColor += weight[1] * textureOffset(bloom, texCoords, ivec2(0, +1)).rgb;
    bloomColor += weight[1] * textureOffset(bloom, texCoords, ivec2(0, -1)).rgb;
    bloomColor += weight[2] * textureOffset(bloom, texCoords, ivec2(0, +2)).rgb;
    bloomColor += weight[2] * textureOffset(bloom, texCoords, ivec2(0, -2)).rgb;
    bloomColor += weight[3] * textureOffset(bloom, texCoords, ivec2(0, +3)).rgb;
    bloomColor += weight[3] * textureOffset(bloom, texCoords, ivec2(0, -3)).rgb;
    bloomColor += weight[4] * textureOffset(bloom, texCoords, ivec2(0, +4)).rgb;
    bloomColor += weight[4] * textureOffset(bloom, texCoords, ivec2(0, -4)).rgb;

    outColor = vec4(bloomColor, 1.0f);
}
