#ifndef MAIN_H
#define MAIN_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <memory>
#include <vector>
#include <chrono>

struct SwapchainProperties {
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    std::uint32_t minImageCount;
    vk::SurfaceTransformFlagBitsKHR transform;
};

class VulkanApplication {

public:
    VulkanApplication();

    void run();

private:
    void recordDrawCommands();
    void drawFrame();

private:
// graphics stuff
    std::unique_ptr<GLFWwindow, void(*)(GLFWwindow*)> m_window;

    vk::UniqueInstance m_instance;
    vk::DispatchLoaderDynamic m_dispatchLoader;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> m_callback;

    vk::UniqueSurfaceKHR m_surface;

    vk::UniqueDevice m_device;
    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;

    SwapchainProperties m_swapchainProps;
    vk::UniqueSwapchainKHR m_swapchain;
    std::vector<vk::Image> m_swapchainImages{};
    std::vector<vk::UniqueImageView> m_swapchainImageViews{};

    vk::UniqueRenderPass m_renderPass;
    vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniquePipeline m_graphicsPipeline;

    std::vector<vk::UniqueFramebuffer> m_framebuffers{};

    vk::UniqueCommandPool m_commandPool;
    std::vector<vk::UniqueCommandBuffer> m_commandBuffers{};

    std::vector<vk::UniqueSemaphore> m_imageAvailableSemaphores{};
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores{};
    std::vector<vk::UniqueFence> m_inFlightFences;

    vk::UniqueBuffer m_vertexBuffer;
    vk::UniqueDeviceMemory m_vertexBufferMemory;
    vk::UniqueBuffer m_indexBuffer;
    vk::UniqueDeviceMemory m_indexBufferMemory;

    std::vector<vk::UniqueBuffer> m_uniformBuffers{};
    std::vector<vk::UniqueDeviceMemory> m_uniformBuffersMemory{};

    vk::UniqueDescriptorPool m_descriptorPool;
    std::vector<vk::UniqueDescriptorSet> m_descriptorSets;

    std::size_t m_currentFrame = 0;

// logic stuff
    std::chrono::system_clock::time_point m_startTime;

    std::size_t m_fpsCounter;
    std::chrono::system_clock::time_point m_lastFpsTimePoint;
};

#endif // MAIN_H
