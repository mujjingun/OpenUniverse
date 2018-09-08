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

std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)> makeWindow(int width, int height);

vk::UniqueInstance makeInstance();

vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> makeDebugMessenger(
    vk::Instance instance, vk::DispatchLoaderDynamic const& dispatchLoader);

vk::UniqueSurfaceKHR makeSurface(GLFWwindow* window, vk::Instance instance);

vk::PhysicalDevice selectPhysicalDevice(vk::Instance instance);

struct QueueFamilyIndices {
    std::uint32_t graphicsFamilyIndex;
    std::uint32_t presentFamilyIndex;
};

QueueFamilyIndices selectQueueFamilyIndices(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

vk::UniqueDevice makeDevice(QueueFamilyIndices queueFamilies, vk::PhysicalDevice physicalDevice);

struct SwapchainProperties {
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    std::uint32_t minImageCount;
    vk::SurfaceTransformFlagBitsKHR transform;
};

SwapchainProperties selectSwapchainProperties(
    vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, GLFWwindow* window);

vk::UniqueSwapchainKHR makeSwapchain(
    SwapchainProperties props, QueueFamilyIndices queueFamilies, vk::SurfaceKHR surface, vk::Device device);

std::vector<vk::Image> retrieveSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain);

vk::UniqueImageView makeImageView(vk::Device device, vk::Image image, vk::Format imageFormat,
    vk::ImageAspectFlags imageType, uint32_t mipLevels);

vk::UniqueRenderPass makeRenderPass(vk::Device device, vk::SampleCountFlagBits sampleCount,
    vk::Format imageFormat, vk::Format depthFormat);

vk::UniquePipelineLayout makePipelineLayout(vk::Device device, vk::DescriptorSetLayout descriptorSetLayout);

vk::UniquePipeline makePipeline(vk::Device device, vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent, vk::RenderPass renderPass, vk::SampleCountFlagBits sampleCount,
    vk::VertexInputBindingDescription bindingDescription,
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions);

std::vector<vk::UniqueFramebuffer> makeFramebuffers(vk::Device device,
    std::vector<vk::UniqueImageView> const& imageViews, vk::ImageView depthImageView, vk::ImageView multiSampleImageView,
    vk::RenderPass renderPass, vk::Extent2D swapChainExtent);

vk::UniqueCommandPool makeCommandPool(vk::Device device, std::uint32_t queueFamilyIndex);

std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(vk::Device device, vk::CommandPool commandPool, std::uint32_t count);

vk::UniqueSemaphore makeSemaphore(vk::Device device);

vk::UniqueFence makeFence(vk::Device device);

vk::UniqueBuffer makeBuffer(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::SharingMode sharingMode = vk::SharingMode::eExclusive);

vk::UniqueDeviceMemory allocateBufferMemory(vk::PhysicalDevice physicalDevice, vk::Device device, vk::Buffer buffer,
    vk::MemoryPropertyFlags properties);

struct BufferObject {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory bufferMemory;
};

BufferObject constructDeviceLocalBuffer(vk::PhysicalDevice physicalDevice, vk::Device device,
    vk::CommandPool commandPool, vk::Queue graphicsQueue, vk::BufferUsageFlags usageFlags,
    const void* bufferData, std::size_t bufferSize);

vk::UniqueDescriptorSetLayout makeDescriptorSetLayout(vk::Device device);

std::chrono::system_clock::time_point getCurrentTimePoint();

vk::UniqueDescriptorPool makeDescriptorPool(vk::Device device, std::uint32_t size);

std::vector<vk::UniqueDescriptorSet> makeDescriptorSets(vk::Device device, vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout, std::uint32_t size);

struct ImageObject {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory imageMemory;
    vk::UniqueImageView imageView;
    vk::Format format;
};

ImageObject makeImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::SampleCountFlagBits numSamples,
    std::uint32_t mipLevels, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);

ImageObject makeTextureImage(vk::PhysicalDevice physicalDevice, vk::Device device, const char* filename,
    vk::CommandPool commandPool, vk::Queue queue);

vk::UniqueSampler makeTextureSampler(vk::Device device);

vk::SampleCountFlagBits getMaxUsableSampleCount(vk::PhysicalDevice physicalDevice);

ImageObject makeDepthImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::CommandPool commandPool,
    vk::Queue queue, vk::Extent2D extent, vk::SampleCountFlagBits sampleCount);

ImageObject makeMultiSampleImage(vk::PhysicalDevice physicalDevice, vk::Device device, VkCommandPool commandPool,
    vk::Queue queue, vk::Format imageFormat, vk::Extent2D extent, vk::SampleCountFlagBits sampleCount);

} // namespace ou

#endif // VULKAN_ROUTINES_H
