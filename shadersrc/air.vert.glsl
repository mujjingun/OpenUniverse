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

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

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

const float pi = acos(-1);
const float thickness = 0.03f;

void main() {
    // generate a sphere

    // int[meridianCount][parallelCount][4]
    int meridianIndex = gl_VertexIndex / (4 * ubo.parallelCount);
    int parallelIndex = gl_VertexIndex % (4 * ubo.parallelCount) / 4;
    int primitiveIndex = gl_VertexIndex % 4;

    // inside the sphere
    if (length(ubo.modelEyePos) - 1.0f < thickness) {
        if ((primitiveIndex == 1) || (primitiveIndex == 2))
            meridianIndex++;
        if ((primitiveIndex == 2) || (primitiveIndex == 3))
            parallelIndex++;
    }
    else {
        if ((primitiveIndex == 1) || (primitiveIndex == 2))
            parallelIndex++;
        if ((primitiveIndex == 2) || (primitiveIndex == 3))
            meridianIndex++;
    }

    vec3 norm = vec3(sin(bounds.mapCenterTheta) * cos(bounds.mapCenterPhi),
                     sin(bounds.mapCenterTheta) * sin(bounds.mapCenterPhi),
                     cos(bounds.mapCenterTheta));
    mat3 rotate = rotationMatrix(normalize(cross(norm, vec3(1, 0, 0))), acos(norm.x));

    float thetaEnd = bounds.mapSpanTheta;
    float theta = mix(0, pi, float(parallelIndex) / ubo.parallelCount);
    float phi = mix(0, 2 * pi, float(meridianIndex) / ubo.meridianCount);

    outPos = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}
