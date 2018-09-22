#ifndef MAIN_H
#define MAIN_H

#include <chrono>
#include <memory>
#include <vector>

#include "vulkan_routines.h"

namespace ou {

// set of elements that need to be recreated when the window gets resized
struct SwapchainObject {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages{};
    std::vector<vk::UniqueImageView> swapchainImageViews{};

    // main render stage
    ImageObject multiSampleImage;
    ImageObject depthImage;
    std::vector<ImageObject> hdrImages{};
    std::vector<ImageObject> bloomImages{};

    vk::UniqueRenderPass renderPass;

    DescriptorSetObject descriptorSet;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline terrainPipeline;
    vk::UniquePipeline atmospherePipeline;
    vk::UniquePipeline numbersPipeline;

    std::vector<vk::UniqueFramebuffer> framebuffers{};

    // present & bloom shader
    DescriptorSetObject bloomHDescriptorSet;
    DescriptorSetObject bloomVDescriptorSet;

    vk::UniquePipelineLayout bloomHPipelineLayout;
    vk::UniquePipeline bloomHPipeline;

    vk::UniquePipelineLayout bloomVPipelineLayout;
    vk::UniquePipeline bloomVPipeline;

    std::vector<vk::UniqueCommandBuffer> commandBuffers{};

    // noise render pass
    std::vector<ImageObject> noiseImages{};
    BufferObject terrain;

    DescriptorSetObject noiseDescriptorSet;
    vk::UniquePipelineLayout noisePipelineLayout;
    vk::UniquePipeline noisePipeline;

    std::vector<vk::UniqueCommandBuffer> noiseCommandBuffers{};

    SwapchainObject() = default;
    SwapchainObject(GraphicsContext const& context, SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain = nullptr);
};

struct Vertex;
struct ModelObject {
    BufferObject vertexBuffer;
    BufferObject indexBuffer;
    std::size_t indexCount;

    ModelObject() = default;
    ModelObject(GraphicsContext const& context, std::vector<Vertex> const& vertices, std::vector<std::uint16_t> const& indices);
};

struct MapBoundsObject {
    float mapCenterTheta;
    float mapCenterPhi;
    float mapSpanTheta;
};

class VulkanApplication {

public:
    VulkanApplication();

    void run();

private:
    void refreshSwapchain();
    void recordDrawCommands();
    void drawFrame();
    void step(std::chrono::duration<double> delta);
    void keyEvent(int key, int scancode, int action, int mods);

private:
    // graphics stuff
    GraphicsContext m_context;
    SwapchainProperties m_swapchainProps;

    SwapchainObject m_swapchain;

    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores{};
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores{};
    std::vector<vk::UniqueFence> m_inFlightFences;
    std::vector<vk::UniqueFence> m_offscreenFences;

    ImageObject m_textureImage;
    vk::UniqueSampler m_sampler;

    std::vector<BufferObject> m_uniformBuffers{};
    std::vector<BufferObject> m_mapBoundsUniformBuffers{};
    std::vector<BufferObject> m_renderMapBoundsUniformBuffers{};
    std::vector<BufferObject> m_numberBuffers{};

    std::size_t m_currentFrame = 0;

    // fps calculation
    std::uint32_t m_currentFps = 0;
    std::size_t m_fpsFrameCounter, m_fpsMeasurementsCount;
    std::chrono::system_clock::time_point m_lastFpsTime;

    // logic stuff
    std::chrono::system_clock::time_point m_lastFrameTime;

    // planet related stuff
    int m_parallelCount, m_meridianCount;
    float m_planetRotateAngle;

    // player camera
    glm::vec3 m_eyePosition;
    glm::vec3 m_lookDirection;
    glm::vec3 m_upDirection;
    bool m_movingForward, m_movingBackward, m_rotatingLeft, m_rotatingRight;

    // control related stuff
    glm::vec2 m_lastCursorPos;
    glm::vec2 m_deltaCursorPos;

    // map related stuff
    bool m_updateOverallmap;
    bool m_updateHeightmap, m_renderingHeightmap;
    std::size_t m_lastRenderedIndex;
    std::vector<MapBoundsObject> m_mapBounds;
};

} // namespace ou

#endif // MAIN_H
