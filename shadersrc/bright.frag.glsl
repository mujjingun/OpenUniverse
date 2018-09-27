#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;

layout(location = 0) out vec4 outColor;

const float exposure = 0.3;

vec3 threshold(vec3 c)
{
    float brightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 color = vec3(1.0) - exp(-c * exposure);
    return mix(vec3(0), color, smoothstep(5.0, 7.0, brightness));
}

void main()
{
    vec3 hdrColor = threshold(subpassLoad(inputColor).rgb);
    outColor = vec4(hdrColor, 1.0f);
}
