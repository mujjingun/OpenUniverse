
set(GLSL_VALIDATOR_EXECUTABLE "${Vulkan_INCLUDE_DIR}/../bin/glslangValidator")

add_custom_target("${PROJECT_NAME}_Shaders")

function(compile_shader SHADER_SOURCE OUTPUT_SPV_NAME)
    set(TARGET_NAME "${PROJECT_NAME}_shadercompile_${SHADER_SOURCE}")
    add_custom_target(${TARGET_NAME}
        COMMAND ${GLSL_VALIDATOR_EXECUTABLE} -V ${SHADER_SOURCE} -o ${OUTPUT_SPV_NAME}
        DEPENDS ${SHADER_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies("${PROJECT_NAME}_Shaders" ${TARGET_NAME})
endfunction(compile_shader)

compile_shader(planet.vert.glsl ../shaders/planet.vert.spv)
compile_shader(planet.tesc.glsl ../shaders/planet.tesc.spv)
compile_shader(planet.tese.glsl ../shaders/planet.tese.spv)
compile_shader(planet.frag.glsl ../shaders/planet.frag.spv)

compile_shader(air.vert.glsl ../shaders/air.vert.spv)
compile_shader(air.tesc.glsl ../shaders/air.tesc.spv)
compile_shader(air.tese.glsl ../shaders/air.tese.spv)
compile_shader(air.frag.glsl ../shaders/air.frag.spv)

compile_shader(noise.vert.glsl ../shaders/noise.vert.spv)
compile_shader(noise.tesc.glsl ../shaders/noise.tesc.spv)
compile_shader(noise.tese.glsl ../shaders/noise.tese.spv)
compile_shader(noise.frag.glsl ../shaders/noise.frag.spv)
