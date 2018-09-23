#ifndef VULKAN_ROUTINES_H
#define VULKAN_ROUTINES_H

#include <chrono>
#include <deque>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

namespace ou {

// TODO: separate glfw utils from vulkan utils
using UniqueWindow = std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)>;
UniqueWindow makeWindow(int width, int height, bool fullscreen);

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
    std::uint32_t imageCount;
    vk::SurfaceTransformFlagBitsKHR transform;
};

struct ImageObject {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::UniqueImageView view;
    vk::Format format;
    vk::Extent2D extent;
    std::uint32_t mipLevels;
    std::uint32_t layerCount;
};

struct BufferObject {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
};

struct DescriptorSetObject {
    vk::UniqueDescriptorSetLayout layout;
    vk::UniqueDescriptorPool pool;
    std::vector<vk::DescriptorSet> sets;
};

class SingleTimeCommandBuffer {
    vk::UniqueCommandBuffer commandBuffer;
    vk::Queue queue;

public:
    SingleTimeCommandBuffer(vk::UniqueCommandBuffer&& commandBuf, vk::Queue const& queue);
    SingleTimeCommandBuffer(SingleTimeCommandBuffer&& other);
    SingleTimeCommandBuffer& operator=(SingleTimeCommandBuffer&& other);
    vk::CommandBuffer operator*() const;
    const vk::CommandBuffer* operator->() const;
    ~SingleTimeCommandBuffer();
};

class GraphicsContext {

public:
    GraphicsContext();
    GraphicsContext(int width, int height, bool fullscreen);

    // getters
    vk::Device device() const;
    GLFWwindow* window() const;
    vk::Queue graphicsQueue() const;
    vk::Queue graphicsQueue2() const;
    vk::Queue presentQueue() const;
    int refreshRate() const;
    vk::Extent2D screenResolution() const;

    // member functions
    bool isFullscreen() const;
    void toggleFullscreenMode() const;

    SwapchainProperties selectSwapchainProperties() const;

    vk::UniqueDescriptorSetLayout makeDescriptorSetLayout(const std::vector<vk::DescriptorType>& types,
        const std::vector<vk::ShaderStageFlags>& stages, const std::vector<uint32_t>& counts) const;
    vk::UniqueDescriptorPool makeDescriptorPool(uint32_t size, std::vector<vk::DescriptorType> const& types) const;
    std::vector<vk::DescriptorSet> makeDescriptorSets(vk::DescriptorPool pool,
        vk::DescriptorSetLayout layout, std::uint32_t size) const;

    DescriptorSetObject makeDescriptorSet(uint32_t size, std::vector<vk::DescriptorType> const& types,
        std::vector<vk::ShaderStageFlags> const& stages, const std::vector<uint32_t>& counts) const;

    vk::UniqueSwapchainKHR makeSwapchain(SwapchainProperties props, vk::SwapchainKHR oldSwapchain = nullptr) const;
    std::vector<vk::Image> retrieveSwapchainImages(vk::SwapchainKHR swapchain) const;

    vk::UniqueImageView makeImageView(vk::Image image, vk::Format imageFormat,
        vk::ImageAspectFlags imageType, uint32_t mipLevels,
        vk::ImageViewType dimensions = vk::ImageViewType::e2D, uint32_t layerCount = 1) const;

    SingleTimeCommandBuffer beginSingleTimeCommands() const;

    ImageObject makeImage(vk::SampleCountFlagBits numSamples, std::uint32_t mipLevels, vk::Extent2D extent, uint32_t layerCount,
        vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlagBits aspect) const;

    ImageObject makeDepthImage(vk::Extent2D extent, vk::SampleCountFlagBits sampleCount) const;

    ImageObject makeMultiSampleImage(vk::Format imageFormat, vk::Extent2D extent, std::uint32_t layerCount,
        vk::SampleCountFlagBits sampleCount) const;

    vk::UniqueRenderPass makeRenderPass(std::vector<vk::SubpassDescription> const& subpasses,
        std::vector<vk::SubpassDependency> const& dependencies,
        std::vector<vk::AttachmentDescription> const& attachments) const;

    vk::UniquePipelineLayout makePipelineLayout(vk::DescriptorSetLayout descriptorSetLayout) const;

    vk::UniquePipeline makePipeline(vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent,
        vk::RenderPass renderPass, uint32_t subpassIndex, vk::SampleCountFlagBits sampleCount,
        const char* vertexShaderFile, const char* fragmentShaderFile, const char* tcShaderFile, const char* teShaderFile,
        const char* geometryShaderFile,
        vk::PrimitiveTopology primitiveType, bool enableBlending,
        bool attachVertexData,
        vk::VertexInputBindingDescription bindingDescription,
        const std::vector<vk::VertexInputAttributeDescription>& attributeDescriptions) const;

    vk::UniquePipeline makeComputePipeline(vk::PipelineLayout pipelineLayout, const char* shaderFile) const;

    std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(std::uint32_t count) const;

    vk::UniqueFramebuffer makeFramebuffer(std::vector<vk::ImageView> imageViews,
        vk::RenderPass renderPass, vk::Extent2D extent, std::size_t layerCount = 1) const;

    std::vector<vk::UniqueSemaphore> makeSemaphores(std::uint32_t count) const;
    std::vector<vk::UniqueFence> makeFences(std::uint32_t count, bool signaled) const;

    vk::UniqueBuffer makeBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
        vk::SharingMode sharingMode = vk::SharingMode::eExclusive) const;

    vk::UniqueDeviceMemory allocateBufferMemory(vk::Buffer buffer, vk::MemoryPropertyFlags properties) const;

    void updateMemory(vk::DeviceMemory memory, void* ptr, std::size_t size) const;

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;

    void copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, std::uint32_t width, std::uint32_t height) const;

    BufferObject constructDeviceLocalBuffer(vk::BufferUsageFlags usageFlags, const void* bufferData, std::size_t bufferSize) const;

    BufferObject makeHostVisibleBuffer(vk::BufferUsageFlags usageFlags, std::size_t bufferSize) const;

    void generateMipmaps(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format,
        vk::Extent2D extent, std::uint32_t mipLevels) const;

    ImageObject makeTextureImage(const char* filename) const;

    vk::UniqueSampler makeTextureSampler(bool unnormalizedCoordinates) const;

    vk::SampleCountFlagBits getMaxUsableSampleCount(std::uint32_t preferredSampleCount) const;

private:
    UniqueWindow m_window;
    const GLFWvidmode* m_videoMode;
    vk::Extent2D m_windowedModeSize;

    vk::UniqueInstance m_instance;

    vk::DispatchLoaderDynamic m_dispatchLoader;

    UniqueDebugMessenger m_messenger;

    vk::UniqueSurfaceKHR m_surface;

    vk::PhysicalDevice m_physicalDevice;

    QueueFamilyIndices m_queueIndices;

    vk::UniqueDevice m_device;

    vk::Queue m_graphicsQueue, m_graphicsQueue2, m_presentQueue;

    vk::UniqueCommandPool m_commandPool;
};

void transitionImageLayout(vk::CommandBuffer commandBuf, vk::Image image, std::uint32_t layerCount,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout, std::uint32_t mipLevels, std::uint32_t baseMipLevel = 0);

} // namespace ou

#endif // VULKAN_ROUTINES_H
