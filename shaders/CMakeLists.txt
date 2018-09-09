
set(GLSL_VALIDATOR_EXECUTABLE "${Vulkan_INCLUDE_DIR}/../bin/glslangValidator")

set(GLSL_SHADER_SRCS
    shader.vert.glsl
    shader.frag.glsl
    shader.tesc.glsl
    shader.tese.glsl
)

add_custom_target("${PROJECT_NAME}_Shaders"
    COMMAND ${GLSL_VALIDATOR_EXECUTABLE} -V ${GLSL_SHADER_SRCS}
    DEPENDS ${GLSL_SHADER_SRCS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
