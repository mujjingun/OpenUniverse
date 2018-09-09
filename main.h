#ifndef MAIN_H
#define MAIN_H

#include <chrono>
#include <memory>
#include <vector>

#include "vulkan_routines.h"

namespace ou {

struct SwapchainObject {
    vk::UniqueSwapchainKHR m_swapchain;
    std::vector<vk::Image> m_swapchainImages{};
    std::vector<vk::UniqueImageView> m_swapchainImageViews{};

    ImageObject m_multiSampleImage;
    ImageObject m_depthImage;

    vk::UniqueRenderPass m_renderPass;

    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniquePipeline m_graphicsPipeline;

    std::vector<vk::UniqueCommandBuffer> m_commandBuffers{};
    std::vector<vk::UniqueFramebuffer> m_framebuffers{};

    SwapchainObject() = default;
    SwapchainObject(GraphicsContext const& context,
        vk::DescriptorSetLayout descriptorSetLayout, SwapchainProperties const& properties);
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

    // set of elements that need to be recreated when the window gets resized
    SwapchainObject m_swapchain;

    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores{};
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores{};
    std::vector<vk::UniqueFence> m_inFlightFences;

    BufferObject m_vertexBuffer;
    BufferObject m_indexBuffer;

    std::vector<vk::UniqueBuffer> m_uniformBuffers{};
    std::vector<vk::UniqueDeviceMemory> m_uniformBuffersMemory{};

    ImageObject m_textureImage;
    vk::UniqueSampler m_sampler;

    std::size_t m_currentFrame = 0;

    // logic stuff
    std::chrono::system_clock::time_point m_startTime;

    std::size_t m_fpsCounter;
    std::chrono::system_clock::time_point m_lastFpsTimePoint;
};

} // namespace ou

#endif // MAIN_H
