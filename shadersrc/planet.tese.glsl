#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (quads, fractional_even_spacing, ccw) in;

layout(set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 eyePos;
    vec4 modelEyePos;
    vec4 lightDir;
    int parallelCount;
    int meridianCount;
    uint noiseIndex;
} ubo;

layout(set = 0, binding = 1, std140) uniform MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
} bounds;

layout(set = 0, binding = 2) uniform sampler2D texSamplers[2];

layout (location = 0) in vec3 inPos[];

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 worldPos;

out gl_PerVertex {
  vec4 gl_Position;
  float gl_PointSize;
  float gl_ClipDistance[];
};

vec3 interpolate3D(vec3 v0, vec3 v1, vec3 v2, vec3 v3)
{
    const vec3 p0 = mix(v0, v3, gl_TessCoord.x);
    const vec3 p1 = mix(v1, v2, gl_TessCoord.x);
    return mix(p0, p1, gl_TessCoord.y);
}

const float pi = acos(-1);

mat3 rotationMatrix(vec3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c        );
}

vec2 getTexCoords(vec3 pos) {
    vec3 mapCenterCart = vec3(sin(bounds.mapCenterTheta) * cos(bounds.mapCenterPhi),
                              sin(bounds.mapCenterTheta) * sin(bounds.mapCenterPhi),
                              cos(bounds.mapCenterTheta));
    mat3 rotate = rotationMatrix(normalize(cross(vec3(1, 0, 0), mapCenterCart)), acos(mapCenterCart.x));

    // (theta, phi) in [0, pi] x [-pi, pi]
    vec3 mapCoords = rotate * pos;
    vec2 sphericalCoords = vec2(acos(mapCoords.z), atan(mapCoords.y, mapCoords.x));
    sphericalCoords.x = (sphericalCoords.x - pi / 2) / bounds.mapSpanTheta / 2 + .5f;
    sphericalCoords.y = sphericalCoords.y / bounds.mapSpanTheta / 2 + .5f;
    vec2 texCoords = vec2(sphericalCoords.x, sphericalCoords.y);
    return texCoords;
}

// The PG subdivided an equilateral triangle into
// smaller triangles and executes the TES for every generated vertex.
void main(void)
{
    const vec3 pos = normalize(interpolate3D(inPos[0], inPos[1], inPos[2], inPos[3]));
    vec2 texCoords = getTexCoords(pos);

    vec4 noiseTex = texture(texSamplers[ubo.noiseIndex], texCoords);
    const float noise = max(0, noiseTex.r);

    vec3 modelPos = pos * (1.0f + vec3(noise * 0.01f));
    worldPos = (ubo.model * vec4(modelPos, 1.0f)).xyz;
    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0f);
    outPos = pos;
}
