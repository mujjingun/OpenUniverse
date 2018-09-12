#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightDir;
    int parallelCount;
    int meridianCount;
} ubo;

layout(location = 0) out vec3 outPos;

mat3 rotationMatrix(vec3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c        );
}

const float pi = 3.1415926536;

void main() {
    // generate a sphere

    // int[meridianCount][parallelCount][4]
    float meridianIndex = float(gl_VertexIndex / (4 * ubo.parallelCount));
    float parallelIndex = float(gl_VertexIndex % (4 * ubo.parallelCount) / 4);
    int primitiveIndex = gl_VertexIndex % 4;

    if ((primitiveIndex == 1) || (primitiveIndex == 2))
        parallelIndex++;
    if ((primitiveIndex == 2) || (primitiveIndex == 3))
        meridianIndex++;

    vec3 modelEyePos = ubo.modelEyePos.xyz;
    vec3 norm = normalize(modelEyePos);
    mat3 rotate = rotationMatrix(normalize(cross(norm, vec3(0, 0, 1))), acos(norm.z));

    float thetaEnd = acos(1.0f / length(modelEyePos));
    float theta = parallelIndex / ubo.parallelCount * thetaEnd;
    float phi = mix(0, 2 * pi, meridianIndex / ubo.meridianCount);

    outPos = rotate * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}
