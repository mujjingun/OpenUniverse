
set(GLSL_VALIDATOR_EXECUTABLE "${Vulkan_INCLUDE_DIR}/../bin/glslangValidator")

add_custom_target("${PROJECT_NAME}_Shaders")

function(compile_shader SHADER_SOURCE OUTPUT_SPV_NAME)
    set(TARGET_NAME "${PROJECT_NAME}_shadercompile_${SHADER_SOURCE}_${OUTPUT_SPV_NAME}")
    add_custom_target(${TARGET_NAME}
        COMMAND ${GLSL_VALIDATOR_EXECUTABLE} -V ${SHADER_SOURCE} -o ${OUTPUT_SPV_NAME}
        DEPENDS ${SHADER_SOURCE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies("${PROJECT_NAME}_Shaders" ${TARGET_NAME})
endfunction(compile_shader)

compile_shader(planet.vert.glsl planet.vert.spv)
compile_shader(planet.tesc.glsl planet.tesc.spv)
compile_shader(planet.tese.glsl planet.tese.spv)
compile_shader(planet.frag.glsl planet.frag.spv)

compile_shader(water.vert.glsl water.vert.spv)
compile_shader(water.tesc.glsl water.tesc.spv)
compile_shader(water.tese.glsl water.tese.spv)
compile_shader(water.frag.glsl water.frag.spv)

compile_shader(air.vert.glsl air.vert.spv)
compile_shader(air.tesc.glsl air.tesc.spv)
compile_shader(air.tese.glsl air.tese.spv)
compile_shader(air.frag.glsl air.frag.spv)
