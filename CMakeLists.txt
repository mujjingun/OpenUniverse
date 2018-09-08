cmake_minimum_required(VERSION 3.10)

project(OpenUniverse)
add_executable(${PROJECT_NAME}
    main.cpp
    vulkan_routines.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS OFF
)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

target_link_libraries(${PROJECT_NAME}
    Vulkan::Vulkan
    glfw
    glm
)

target_include_directories(${PROJECT_NAME} PRIVATE libraries)

add_subdirectory(shaders)
add_dependencies(${PROJECT_NAME} "${PROJECT_NAME}_Shaders")
