#ifndef MAIN_H
#define MAIN_H

#include <chrono>
#include <memory>
#include <vector>

#include "vulkan_routines.h"

namespace ou {

struct SwapchainObject {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages{};
    std::vector<vk::UniqueImageView> swapchainImageViews{};

    ImageObject multiSampleImage;
    ImageObject depthImage;

    vk::UniqueRenderPass renderPass;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;

    std::vector<vk::UniqueCommandBuffer> commandBuffers{};
    std::vector<vk::UniqueFramebuffer> framebuffers{};

    SwapchainObject() = default;
    SwapchainObject(GraphicsContext const& context, vk::DescriptorSetLayout descriptorSetLayout,
                    SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain = nullptr);
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

    ImageObject m_textureImage;
    vk::UniqueSampler m_sampler;

    std::vector<vk::UniqueBuffer> m_uniformBuffers{};
    std::vector<vk::UniqueDeviceMemory> m_uniformBuffersMemory{};

    std::size_t m_currentFrame = 0;

    // logic stuff
    std::chrono::system_clock::time_point m_startTime;

    std::size_t m_fpsCounter;
    std::chrono::system_clock::time_point m_lastFpsTimePoint;
};

} // namespace ou

#endif // MAIN_H
