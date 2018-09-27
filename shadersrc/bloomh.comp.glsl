#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define WORKGROUP_SIZE 32
layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

layout(binding = 0, rgba8) uniform writeonly image2D result;
layout(binding = 1, rgba8) uniform readonly image2D source;

const float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main()
{
    const ivec2 size = imageSize(source);
    const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

    if (pos.x >= size.x || pos.y >= size.y) {
        return;
    }

    // horizontal pass
    vec4 avg = weight[0] * imageLoad(source, pos + ivec2(0, 0));
    avg += weight[1] * imageLoad(source, pos + ivec2(1, 0));
    avg += weight[1] * imageLoad(source, pos + ivec2(-1, 0));
    avg += weight[2] * imageLoad(source, pos + ivec2(2, 0));
    avg += weight[2] * imageLoad(source, pos + ivec2(-2, 0));
    avg += weight[3] * imageLoad(source, pos + ivec2(3, 0));
    avg += weight[3] * imageLoad(source, pos + ivec2(-3, 0));
    avg += weight[4] * imageLoad(source, pos + ivec2(4, 0));
    avg += weight[4] * imageLoad(source, pos + ivec2(-4, 0));

    imageStore(result, pos, avg);
}
