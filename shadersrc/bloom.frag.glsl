#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform sampler2D image;

layout(location = 0) out vec4 outColor;

layout (location = 0) in vec2 texCoords;

void main()
{
    const float exposure = 1.0;
    const float gamma = 2.2;
    vec3 hdrColor = texture(image, texCoords).rgb;
    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    //mapped = pow(mapped, vec3(1.0 / gamma));
    outColor = vec4(mapped, 1.0f);
}
