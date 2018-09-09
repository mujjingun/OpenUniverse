#ifndef VULKAN_ROUTINES_H
#define VULKAN_ROUTINES_H

#include <chrono>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace ou {

using UniqueWindow = std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)>;
UniqueWindow makeWindow(int width, int height);

vk::UniqueInstance makeInstance();

using UniqueDebugMessenger = vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>;
UniqueDebugMessenger makeDebugMessenger(vk::Instance instance, vk::DispatchLoaderDynamic const& dispatchLoader);

vk::UniqueSurfaceKHR makeSurface(GLFWwindow* window, vk::Instance instance);

vk::PhysicalDevice selectPhysicalDevice(vk::Instance instance);

struct QueueFamilyIndices {
    std::uint32_t graphics;
    std::uint32_t presentation;
};

QueueFamilyIndices selectQueueFamilyIndices(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

vk::UniqueDevice makeDevice(QueueFamilyIndices queueFamilies, vk::PhysicalDevice physicalDevice);

vk::UniqueCommandPool makeCommandPool(vk::Device device, std::uint32_t queueFamilyIndex);

struct SwapchainProperties {
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    std::uint32_t minImageCount;
    vk::SurfaceTransformFlagBitsKHR transform;
};

struct ImageObject {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::UniqueImageView view;
    vk::Format format;
};

struct BufferObject {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory bufferMemory;
};

class GraphicsContext {

public:
    GraphicsContext();

    // getters
    vk::Device device() const;
    GLFWwindow *window() const;
    vk::Queue graphicsQueue() const;
    vk::Queue presentQueue() const;

    // member functions
    SwapchainProperties selectSwapchainProperties() const;

    vk::UniqueDescriptorSetLayout makeDescriptorSetLayout() const;
    vk::UniqueDescriptorPool makeDescriptorPool(std::uint32_t size) const;
    std::vector<vk::DescriptorSet> makeDescriptorSets(vk::DescriptorPool pool,
        vk::DescriptorSetLayout layout, std::uint32_t size) const;

    vk::UniqueSwapchainKHR makeSwapchain(SwapchainProperties props) const;
    std::vector<vk::Image> retrieveSwapchainImages(vk::SwapchainKHR swapchain) const;

    vk::UniqueImageView makeImageView(vk::Image image, vk::Format imageFormat,
        vk::ImageAspectFlags imageType, uint32_t mipLevels) const;

    ImageObject makeImage(vk::SampleCountFlagBits numSamples, std::uint32_t mipLevels, vk::Extent2D extent,
        vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlagBits aspect) const;

    ImageObject makeDepthImage(vk::Extent2D extent, vk::SampleCountFlagBits sampleCount) const;

    ImageObject makeMultiSampleImage(vk::Format imageFormat, vk::Extent2D extent, vk::SampleCountFlagBits sampleCount) const;

    vk::UniqueRenderPass makeRenderPass(vk::SampleCountFlagBits sampleCount, vk::Format imageFormat,
        vk::Format depthFormat) const;

    vk::UniquePipelineLayout makePipelineLayout(vk::DescriptorSetLayout descriptorSetLayout) const;

    vk::UniquePipeline makePipeline(vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent, vk::RenderPass renderPass,
        vk::SampleCountFlagBits sampleCount, vk::VertexInputBindingDescription bindingDescription,
        std::vector<vk::VertexInputAttributeDescription> const& attributeDescriptions) const;

    std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(std::uint32_t count) const;

    std::vector<vk::UniqueFramebuffer> makeFramebuffers(std::vector<vk::UniqueImageView> const& imageViews,
        vk::ImageView depthImageView, vk::ImageView multiSampleImageView,
        vk::RenderPass renderPass, vk::Extent2D swapChainExtent) const;

    vk::UniqueSemaphore makeSemaphore() const;
    vk::UniqueFence makeFence() const;

    vk::UniqueBuffer makeBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
        vk::SharingMode sharingMode = vk::SharingMode::eExclusive) const;

    vk::UniqueDeviceMemory allocateBufferMemory(vk::Buffer buffer, vk::MemoryPropertyFlags properties) const;

    BufferObject constructDeviceLocalBuffer(vk::BufferUsageFlags usageFlags, const void* bufferData, std::size_t bufferSize) const;

    ImageObject makeTextureImage(const char* filename) const;

    vk::UniqueSampler makeTextureSampler() const;

    vk::SampleCountFlagBits getMaxUsableSampleCount(uint32_t preferredSampleCount) const;

private:
    UniqueWindow m_window;

    vk::UniqueInstance m_instance;

    vk::DispatchLoaderDynamic m_dispatchLoader;

    UniqueDebugMessenger m_messenger;

    vk::UniqueSurfaceKHR m_surface;

    vk::PhysicalDevice m_physicalDevice;

    QueueFamilyIndices m_queueIndices;

    vk::UniqueDevice m_device;

    vk::Queue m_graphicsQueue, m_presentQueue;

    vk::UniqueCommandPool m_commandPool;
};

std::chrono::system_clock::time_point getCurrentTimePoint();

} // namespace ou

#endif // VULKAN_ROUTINES_H