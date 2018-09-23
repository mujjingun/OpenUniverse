#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D image;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

const float exposure = 0.3;
const float gamma = 2.2;
const float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

vec3 threshold(vec3 c)
{
    float brightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 5.0) {
        return c;
    }
    else {
        return vec3(0);
    }
}

void main()
{
    vec3 hdrColor = weight[0] * threshold(texture(image, texCoords).rgb);
    hdrColor += weight[1] * threshold(textureOffset(image, texCoords, ivec2(+1, 0)).rgb);
    hdrColor += weight[1] * threshold(textureOffset(image, texCoords, ivec2(-1, 0)).rgb);
    hdrColor += weight[2] * threshold(textureOffset(image, texCoords, ivec2(+2, 0)).rgb);
    hdrColor += weight[2] * threshold(textureOffset(image, texCoords, ivec2(-2, 0)).rgb);
    hdrColor += weight[3] * threshold(textureOffset(image, texCoords, ivec2(+3, 0)).rgb);
    hdrColor += weight[3] * threshold(textureOffset(image, texCoords, ivec2(-3, 0)).rgb);
    hdrColor += weight[4] * threshold(textureOffset(image, texCoords, ivec2(+4, 0)).rgb);
    hdrColor += weight[4] * threshold(textureOffset(image, texCoords, ivec2(-4, 0)).rgb);

    outColor = vec4(hdrColor, 1.0f);
}
