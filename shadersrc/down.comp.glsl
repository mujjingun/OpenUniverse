#version 440
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define WORKGROUP_SIZE 32
layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

layout(binding = 0, rgba8) uniform readonly image2D hdr;
layout(binding = 1, rgba8) uniform image2D bloom;

shared vec4 colors[WORKGROUP_SIZE][WORKGROUP_SIZE];

void main()
{
    ivec2 size = imageSize(hdr) / 2;
    vec4 color00, color10, color01, color11, avg;
    const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 readPos = pos * 2;
    ivec2 groupStart = ivec2(gl_WorkGroupID.x * WORKGROUP_SIZE, gl_WorkGroupID.y * WORKGROUP_SIZE);
    ivec2 localPos = ivec2(gl_LocalInvocationID.xy);

    if (pos.x >= size.x || pos.y >= size.y) {
        return;
    }

    color00 = imageLoad(hdr, readPos + ivec2(0, 0));
    color10 = imageLoad(hdr, readPos + ivec2(1, 0));
    color01 = imageLoad(hdr, readPos + ivec2(0, 1));
    color11 = imageLoad(hdr, readPos + ivec2(1, 1));
    avg = (color00 + color10 + color01 + color11) / 4;

    colors[localPos.x][localPos.y] = avg;

    imageStore(bloom, groupStart + localPos, avg);

    ivec2 offset = ivec2(size.x + 5, 0);
    for (int i = 1; i < 3; ++i) {
        size /= 2;
        groupStart /= 2;
        localPos /= 2;

        barrier();

        color00 = colors[localPos.x * 2 + 0][localPos.y * 2 + 0];
        color10 = colors[localPos.x * 2 + 1][localPos.y * 2 + 0];
        color01 = colors[localPos.x * 2 + 0][localPos.y * 2 + 1];
        color11 = colors[localPos.x * 2 + 1][localPos.y * 2 + 1];
        avg = (color00 + color10 + color01 + color11);

        barrier();

        colors[localPos.x][localPos.y] = avg;

        imageStore(bloom, offset + groupStart + localPos, avg);

        offset.y += size.y + 5;
    }
}
