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

    ImageObject multiSampleImage;
    ImageObject depthImage;
    ImageObject noiseImage;
    ImageObject noiseDepthImage;
    ImageObject noiseMultiSampleImage;

    vk::UniqueRenderPass renderPass;
    vk::UniqueRenderPass noiseRenderPass;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline terrainPipeline;
    vk::UniquePipeline atmospherePipeline;
    vk::UniquePipeline noisePipeline;

    std::vector<vk::UniqueCommandBuffer> commandBuffers{};
    vk::UniqueCommandBuffer noiseCommandBuffer;

    std::vector<vk::UniqueFramebuffer> framebuffers{};
    vk::UniqueFramebuffer noiseFramebuffer;

    SwapchainObject() = default;
    SwapchainObject(GraphicsContext const& context, vk::DescriptorSetLayout descriptorSetLayout,
                    SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain = nullptr);
};

struct Vertex;

struct ModelObject {
    BufferObject vertexBuffer;
    BufferObject indexBuffer;
    std::size_t indexCount;

    ModelObject() = default;
    ModelObject(GraphicsContext const& context, std::vector<Vertex> const& vertices, std::vector<std::uint16_t> const& indices);
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

    vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
    vk::UniqueDescriptorPool m_descriptorPool;
    std::vector<vk::DescriptorSet> m_descriptorSets;

    SwapchainObject m_swapchain;

    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores{};
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores{};
    vk::UniqueSemaphore m_offscreenPassSemaphore;
    std::vector<vk::UniqueFence> m_inFlightFences;

    ImageObject m_textureImage;
    vk::UniqueSampler m_sampler;

    std::vector<BufferObject> m_uniformBuffers{};
    BufferObject m_noiseUniformBuffer;

    std::size_t m_currentFrame = 0;

    // fps calculation
    double m_averageFps;
    std::size_t m_fpsFrameCounter, m_fpsMeasurementsCount;
    std::chrono::system_clock::time_point m_lastFpsTime;

    // logic stuff
    std::chrono::system_clock::time_point m_lastFrameTime;

    // planet related stuff
    int m_parallelCount, m_meridianCount;
    bool m_renderHeightmap;
    float m_planetRotateAngle;

    // player camera
    glm::vec3 m_eyePosition;
    glm::vec3 m_lookDirection;
    glm::vec3 m_upDirection;
    bool m_movingForward, m_movingBackward, m_rotatingLeft, m_rotatingRight;

    glm::vec2 m_lastCursorPos;
    glm::vec2 m_deltaCursorPos;
};

} // namespace ou

#endif // MAIN_H
