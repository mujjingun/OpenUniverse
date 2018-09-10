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

    vk::UniqueRenderPass renderPass;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline terrainPipeline;
    vk::UniquePipeline oceanPipeline;
    vk::UniquePipeline atmospherePipeline;

    std::vector<vk::UniqueCommandBuffer> commandBuffers{};
    std::vector<vk::UniqueFramebuffer> framebuffers{};

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
    std::vector<vk::UniqueFence> m_inFlightFences;

    ImageObject m_textureImage;
    vk::UniqueSampler m_sampler;

    ModelObject m_model;

    std::vector<vk::UniqueBuffer> m_uniformBuffers{};
    std::vector<vk::UniqueDeviceMemory> m_uniformBuffersMemory{};

    std::size_t m_currentFrame = 0;

    // logic stuff
    std::chrono::system_clock::time_point m_startTime;
    std::chrono::system_clock::time_point m_lastFrameTime;

    // fps calculation
    double m_averageFps;
    std::size_t m_fpsFrameCounter, m_fpsMeasurementsCount;
    std::chrono::system_clock::time_point m_lastFpsTime;
};

} // namespace ou

#endif // MAIN_H
