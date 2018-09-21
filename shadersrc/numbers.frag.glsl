#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 3, std140) uniform Digits {
    uint n;
} digits[3];

layout(location = 0) out vec4 outColor;

#define CHAR_WIDTH 10
#define CHAR_HEIGHT 20
#define SPACE 5

float figures[10][5][5] = float[][][](
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 0, 0),
        float[](0, 0, 1, 0, 0),
        float[](0, 0, 1, 0, 0),
        float[](0, 0, 1, 0, 0),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 0),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 0, 0, 0, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](0, 0, 0, 0, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 0),
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 0),
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](0, 0, 0, 1, 0),
        float[](0, 0, 1, 0, 0),
        float[](0, 0, 1, 0, 0)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    ),
    float[][](
        float[](1, 1, 1, 1, 1),
        float[](1, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1),
        float[](0, 0, 0, 0, 1),
        float[](1, 1, 1, 1, 1)
    )
);

void main() {
    vec2 coord = gl_FragCoord.xy;

    if (coord.x > SPACE && coord.x < (CHAR_WIDTH + SPACE) * 3 + SPACE && coord.y < CHAR_HEIGHT) {
        coord.x = (coord.x - SPACE) / (CHAR_WIDTH + SPACE);
        coord.y /= CHAR_HEIGHT;

        int index = int(coord.x);
        coord.x = mod(coord.x, 1) * 1.2;

        uint digit = digits[index].n;
        int x = int(coord.x * 5);
        int y = int(coord.y * 5);

        if (x >= 5) discard;

        outColor = vec4(vec3(figures[digit][y][x]), 1.0f);
        return;
    }

    discard;
}
