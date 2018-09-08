#include "main.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const bool enableValidationLayers =
#ifdef NDEBUG
    false;
#else
    true;
#endif

const std::vector<const char*> requiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const std::size_t maxFramesInFlight = 2;

std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)> makeWindow(int width, int height)
{
    // initialize glfw context
    glfwInit();

    // tell glfw to not use OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // disable window resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // create window
    GLFWwindow* window = glfwCreateWindow(width, height, "OpenUniverse on Vulkan", nullptr, nullptr);

    std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)> uniqueWindow(window, [](GLFWwindow* window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    });

    return uniqueWindow;
}

vk::UniqueInstance makeInstance()
{
    // create a vulkan instance
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "OpenUniverse";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "OpenUniverse Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    vk::InstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.setPApplicationInfo(&appInfo);

    // get extensions required for glfw
    std::uint32_t glfwExtensionCount;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> requiredInstanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    // add required extensions for validation layer
    if (enableValidationLayers) {
        requiredInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // check for vulkan extensions
    auto availableInstanceExtensions = vk::enumerateInstanceExtensionProperties();

    std::cout << "Available instance extensions:" << std::endl;
    for (const auto& extension : availableInstanceExtensions) {
        std::cout << "\t" << extension.extensionName << std::endl;
    }

    // check if all glfw extensions are supported
    std::cout << "Required instance extensions:" << std::endl;
    for (const char* requiredExtension : requiredInstanceExtensions) {

        bool supported = std::any_of(availableInstanceExtensions.begin(), availableInstanceExtensions.end(),
            [&](vk::ExtensionProperties const& extension) {
                return std::strcmp(extension.extensionName, requiredExtension) != 0;
            });

        std::cout << "\t" << requiredExtension
                  << " supported=" << std::boolalpha << supported << std::endl;

        if (!supported) {
            throw std::runtime_error("required extension not supported");
        }
    };

    instanceCreateInfo.enabledExtensionCount = static_cast<std::uint32_t>(requiredInstanceExtensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = requiredInstanceExtensions.data();

    // Check layers
    auto availableLayers = vk::enumerateInstanceLayerProperties();

    std::cout << "Available layers:" << std::endl;
    for (const auto& layer : availableLayers) {
        std::cout << "\t" << layer.layerName << std::endl;
    }

    if (enableValidationLayers) {
        std::cout << "Using validation layers:" << std::endl;
        for (const char* validationLayer : validationLayers) {
            bool supported = std::any_of(availableLayers.begin(), availableLayers.end(),
                [&](VkLayerProperties const& layer) {
                    return std::strcmp(layer.layerName, validationLayer) != 0;
                });

            std::cout << "\t" << validationLayer
                      << " supported=" << std::boolalpha << supported << std::endl;

            if (!supported) {
                throw std::runtime_error("validation layer requested but not found");
            }
        }

        // number of global validation layers
        instanceCreateInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());

        // validation layer names to use
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();

    } else {
        std::cout << "NOT using validation layers.";

        // number of global validation layers
        instanceCreateInfo.enabledLayerCount = 0;
    }

    return vk::createInstanceUnique(instanceCreateInfo);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /* pUserData */)
{
    switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        std::clog << "[ERROR]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        std::clog << "[WARN]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        std::clog << "[VERBOSE]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    default:
        std::clog << "[INFO]";
    }

    switch (messageType) {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
        std::clog << "[validation]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
        std::clog << "[performance]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
    default:
        std::clog << "[general]";
    }

    std::clog << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> makeDebugMessenger(
    vk::Instance instance, vk::DispatchLoaderDynamic const& dispatchLoader)
{
    // register debug callback
    if (enableValidationLayers) {
        vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
        createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr;

        return instance.createDebugUtilsMessengerEXTUnique(createInfo, nullptr, dispatchLoader);
    }

    return vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>();
}

vk::UniqueSurfaceKHR makeSurface(GLFWwindow* window, vk::Instance instance)
{
    VkSurfaceKHR surface;

    // create window surface
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface");
    }

    vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic> surfaceDeleter(instance);
    return vk::UniqueSurfaceKHR(surface, surfaceDeleter);
}

vk::PhysicalDevice selectPhysicalDevice(vk::Instance instance)
{
    // search for devices
    auto devices = instance.enumeratePhysicalDevices();

    std::cout << "Available devices:" << std::endl;
    auto foundPhysicalDeviceIt = std::find_if(devices.begin(), devices.end(),
        [&](vk::PhysicalDevice device) {
            vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
            vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();

            bool isSuitable = deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
                && deviceFeatures.geometryShader;

            std::cout << "\t" << deviceProperties.deviceName
                      << " suitable=" << std::boolalpha << isSuitable << std::endl;

            return isSuitable;
        });

    if (foundPhysicalDeviceIt == devices.end()) {
        throw std::runtime_error("no suitable device found");
    }

    return *foundPhysicalDeviceIt;
}

struct QueueFamilyIndices {
    std::uint32_t graphicsFamilyIndex;
    std::uint32_t presentFamilyIndex;
};

QueueFamilyIndices selectQueueFamilyIndices(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
{
    QueueFamilyIndices indices;

    // retrieve queue family info
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    auto graphicsFamilyIt = std::find_if(queueFamilies.begin(), queueFamilies.end(),
        [&](vk::QueueFamilyProperties const& queueFamily) -> bool {
            return queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics;
        });

    if (graphicsFamilyIt == queueFamilies.end()) {
        throw std::runtime_error("no graphics queue family found");
    }

    indices.graphicsFamilyIndex = static_cast<std::uint32_t>(graphicsFamilyIt - queueFamilies.begin());

    auto presentFamilyIt = std::find_if(queueFamilies.begin(), queueFamilies.end(),
        [&](vk::QueueFamilyProperties const& queueFamily) -> bool {
            const std::uint32_t index = static_cast<std::uint32_t>(&queueFamily - queueFamilies.data());
            vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(index, surface);
            return queueFamily.queueCount > 0 && presentSupport;
        });

    if (presentFamilyIt == queueFamilies.end()) {
        throw std::runtime_error("no presentation queue family found");
    }

    indices.presentFamilyIndex = static_cast<std::uint32_t>(presentFamilyIt - queueFamilies.begin());

    return indices;
}

vk::UniqueDevice makeDevice(QueueFamilyIndices queueFamilies, vk::PhysicalDevice physicalDevice)
{
    // check device for swap chain support
    auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    std::cout << "Available device extensions: " << std::endl;
    for (const auto& extension : availableDeviceExtensions) {
        std::cout << "\t" << extension.extensionName << std::endl;
    }

    std::cout << "Required device extensions: " << std::endl;
    for (const char* requiredDeviceExtension : requiredDeviceExtensions) {
        bool supported = std::any_of(availableDeviceExtensions.begin(), availableDeviceExtensions.end(),
            [&](vk::ExtensionProperties const& extension) {
                return strcmp(extension.extensionName, requiredDeviceExtension) == 0;
            });

        std::cout << "\t" << requiredDeviceExtension
                  << " supported=" << std::boolalpha << supported << std::endl;

        if (!supported) {
            throw std::runtime_error("required device extension not found");
        }
    }

    // create logical device
    std::set<std::uint32_t> uniqueQueueFamilyIndices = {
        queueFamilies.graphicsFamilyIndex, queueFamilies.presentFamilyIndex
    };
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    for (std::uint32_t uniqueQueueFamilyIndex : uniqueQueueFamilyIndices) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = uniqueQueueFamilyIndex;

        std::array<float, 1> queuePriorities = { { 1.0f } };
        queueCreateInfo.queueCount = queuePriorities.size();
        queueCreateInfo.pQueuePriorities = queuePriorities.data();

        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.enabledExtensionCount = static_cast<std::uint32_t>(requiredDeviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();

    deviceCreateInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    if (enableValidationLayers) {
        deviceCreateInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
        deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        deviceCreateInfo.enabledLayerCount = 0;
    }

    return physicalDevice.createDeviceUnique(deviceCreateInfo);
}

SwapchainProperties selectSwapchainProperties(
    vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, GLFWwindow* window)
{
    // set up the swap chain
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);

    auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    auto surfacePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);

    if (surfaceFormats.empty() || surfacePresentModes.empty()) {
        throw std::runtime_error("swap chain not supported");
    }

    SwapchainProperties properties;

    // determine swap surface format
    properties.surfaceFormat = [&]() -> vk::SurfaceFormatKHR {

        // no preferred format, only format is undefined
        if (surfaceFormats.size() == 1 && surfaceFormats[0].format == vk::Format::eUndefined) {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        // find suitable format
        for (const auto& availableFormat : surfaceFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8Unorm
                && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        // fine, just use the first one in the list
        return surfaceFormats[0];
    }();

    // determine presentation mode
    properties.presentMode = [&]() {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& availablePresentMode : surfacePresentModes) {

            // use triple buffering if possible
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }

        return bestMode;
    }();

    // choose swap surface dimensions
    properties.extent = [&]() {
        if (surfaceCapabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return surfaceCapabilities.currentExtent;
        } else {
            int windowWidth, windowHeight;
            glfwGetFramebufferSize(window, &windowWidth, &windowHeight);

            vk::Extent2D actualExtent{};
            actualExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
                std::min(surfaceCapabilities.maxImageExtent.width, static_cast<std::uint32_t>(windowWidth)));
            actualExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
                std::min(surfaceCapabilities.maxImageExtent.height, static_cast<std::uint32_t>(windowHeight)));

            return actualExtent;
        }
    }();

    // choose number of images to use
    properties.minImageCount = [&]() {
        std::uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
            imageCount = surfaceCapabilities.maxImageCount;
        }
        return imageCount;
    }();

    return properties;
}

vk::UniqueSwapchainKHR makeSwapchain(
    SwapchainProperties props, QueueFamilyIndices queueFamilies, vk::SurfaceKHR surface, vk::Device device)
{
    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.surface = surface;
    swapChainCreateInfo.minImageCount = props.minImageCount;
    swapChainCreateInfo.imageFormat = props.surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = props.surfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = props.extent;
    swapChainCreateInfo.presentMode = props.presentMode;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    if (queueFamilies.graphicsFamilyIndex != queueFamilies.presentFamilyIndex) {
        // image needs to shared by two queues
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;

        std::array<std::uint32_t, 2> queueFamilyIndices = {
            { queueFamilies.graphicsFamilyIndex, queueFamilies.presentFamilyIndex }
        };
        swapChainCreateInfo.queueFamilyIndexCount = queueFamilyIndices.size();
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    } else {
        // image only needs to be accessed by one queue
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
        swapChainCreateInfo.queueFamilyIndexCount = 0;
        swapChainCreateInfo.pQueueFamilyIndices = nullptr;
    }

    swapChainCreateInfo.preTransform = props.transform;
    swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapChainCreateInfo.clipped = VK_TRUE;
    swapChainCreateInfo.oldSwapchain = nullptr;

    return device.createSwapchainKHRUnique(swapChainCreateInfo);
}

std::vector<vk::Image> retrieveSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain)
{
    // retrieve swap chain image handles
    return device.getSwapchainImagesKHR(swapchain);
}

vk::UniqueImageView makeImageView(vk::Device device, vk::Image image, vk::Format imageFormat,
    vk::ImageAspectFlags imageType = vk::ImageAspectFlagBits::eColor)
{
    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = imageFormat;

    // don't shuffle components around
    imageViewCreateInfo.components.r = vk::ComponentSwizzle::eIdentity;
    imageViewCreateInfo.components.g = vk::ComponentSwizzle::eIdentity;
    imageViewCreateInfo.components.b = vk::ComponentSwizzle::eIdentity;
    imageViewCreateInfo.components.a = vk::ComponentSwizzle::eIdentity;

    // no mipmapping
    imageViewCreateInfo.subresourceRange.aspectMask = imageType;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    return device.createImageViewUnique(imageViewCreateInfo);
}

std::vector<char> readFile(const char* fileName)
{
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("filed to open file");
    }

    std::size_t fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    return buffer;
};

vk::UniqueShaderModule createShaderModule(vk::Device device, std::vector<char> const& code)
{
    vk::ShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.codeSize = code.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

    return device.createShaderModuleUnique(shaderModuleCreateInfo);
}

vk::UniqueRenderPass makeRenderPass(vk::Device device, vk::Format imageFormat, vk::Format depthFormat)
{
    std::array<vk::AttachmentDescription, 2> attachments;

    // color attachment
    attachments[0].format = imageFormat;
    attachments[0].samples = vk::SampleCountFlagBits::e1;
    attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[0].initialLayout = vk::ImageLayout::eUndefined;
    attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

    // depth attachment
    attachments[1].format = depthFormat;
    attachments[1].samples = vk::SampleCountFlagBits::e1;
    attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].initialLayout = vk::ImageLayout::eUndefined;
    attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    std::array<vk::AttachmentReference, 1> colorAttachmentRefs{};
    // layout(location = 0) out vec4 outColor
    colorAttachmentRefs[0].attachment = 0;
    colorAttachmentRefs[0].layout = vk::ImageLayout::eColorAttachmentOptimal;

    // only 1 depth attachment allowed
    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = colorAttachmentRefs.size();
    subpass.pColorAttachments = colorAttachmentRefs.data();
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    // wait until the image is loaded
    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = {};
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    return device.createRenderPassUnique(renderPassInfo);
}

vk::UniquePipelineLayout makePipelineLayout(vk::Device device, vk::DescriptorSetLayout descriptorSetLayout)
{
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    return device.createPipelineLayoutUnique(pipelineLayoutInfo);
}

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

vk::UniquePipeline makePipeline(
    vk::Device device, vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent, vk::RenderPass renderPass)
{
    // make shaders
    vk::UniqueShaderModule vertShaderModule = createShaderModule(device, readFile("shaders/vert.spv"));
    vk::UniqueShaderModule fragShaderModule = createShaderModule(device, readFile("shaders/frag.spv"));

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{};

    shaderStages[0].stage = vk::ShaderStageFlagBits::eVertex;
    shaderStages[0].module = vertShaderModule.get();
    shaderStages[0].pName = "main";

    shaderStages[1].stage = vk::ShaderStageFlagBits::eFragment;
    shaderStages[1].module = fragShaderModule.get();
    shaderStages[1].pName = "main";

    // build the pipeline
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = swapExtent.width;
    viewport.height = swapExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor{};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = swapExtent;

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR
        | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne; // Optional
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero; // Optional
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero; // Optional
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd; // Optional

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;

    pipelineInfo.layout = pipelineLayout;

    // use this pipeline in the render pass
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = -1;

    return device.createGraphicsPipelineUnique(nullptr, pipelineInfo);
}

std::vector<vk::UniqueFramebuffer> makeFramebuffers(vk::Device device,
    std::vector<vk::UniqueImageView> const& imageViews, vk::ImageView depthImageView,
    vk::RenderPass renderPass, vk::Extent2D swapChainExtent)
{
    std::vector<vk::UniqueFramebuffer> framebuffers;

    for (auto const& uniqueImageView : imageViews) {
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderPass;

        std::array<vk::ImageView, 2> attachments = { { uniqueImageView.get(), depthImageView } };
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();

        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        framebuffers.push_back(device.createFramebufferUnique(framebufferInfo));
    }

    return framebuffers;
}

vk::UniqueCommandPool makeCommandPool(vk::Device device, std::uint32_t queueFamilyIndex)
{
    vk::CommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    poolCreateInfo.flags = {};

    return device.createCommandPoolUnique(poolCreateInfo);
}

std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(vk::Device device, vk::CommandPool commandPool, std::uint32_t count)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = count;

    return device.allocateCommandBuffersUnique(allocInfo);
}

vk::UniqueSemaphore makeSemaphore(vk::Device device)
{
    vk::SemaphoreCreateInfo semaphoreInfo{};
    return device.createSemaphoreUnique(semaphoreInfo);
}

vk::UniqueFence makeFence(vk::Device device)
{
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    return device.createFenceUnique(fenceInfo);
}

std::uint32_t selectMemoryType(vk::PhysicalDevice physicalDevice,
    std::uint32_t typeFilter, vk::MemoryPropertyFlags requiredProperties)
{
    vk::PhysicalDeviceMemoryProperties memProperties;
    memProperties = physicalDevice.getMemoryProperties();

    std::bitset<32> typeFilterBits(typeFilter);

    for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if (typeFilterBits.test(i)
            && (memProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties) {
            return i;
        }
    }

    throw std::runtime_error("no suitable memory type found");
}

vk::UniqueBuffer makeBuffer(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::SharingMode sharingMode = vk::SharingMode::eExclusive)
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = sharingMode;

    return device.createBufferUnique(bufferInfo);
}

vk::UniqueDeviceMemory allocateBufferMemory(vk::PhysicalDevice physicalDevice, vk::Device device, vk::Buffer buffer,
    vk::MemoryPropertyFlags properties)
{
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    vk::UniqueDeviceMemory bufferMemory = device.allocateMemoryUnique(allocInfo);

    device.bindBufferMemory(buffer, bufferMemory.get(), 0);

    return bufferMemory;
}

vk::UniqueCommandBuffer beginSingleTimeCommands(vk::Device device, vk::CommandPool pool)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;

    vk::UniqueCommandBuffer commandBuffer = std::move(device.allocateCommandBuffersUnique(allocInfo)[0]);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer->begin(beginInfo);

    return commandBuffer;
}

void submitSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::Queue queue)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    queue.submit({ submitInfo }, nullptr);
    queue.waitIdle();
}

void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size,
    vk::Device device, vk::CommandPool pool, vk::Queue queue)
{
    vk::UniqueCommandBuffer commandBuffer = beginSingleTimeCommands(device, pool);

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer->copyBuffer(srcBuffer, dstBuffer, { copyRegion });

    submitSingleTimeCommands(commandBuffer.get(), queue);
}

void copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, std::uint32_t width, std::uint32_t height,
    vk::Device device, vk::CommandPool pool, vk::Queue queue)
{
    vk::UniqueCommandBuffer commandBuffer = beginSingleTimeCommands(device, pool);

    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset.x = 0;
    region.imageOffset.y = 0;
    region.imageOffset.z = 0;

    region.imageExtent.width = width;
    region.imageExtent.height = height;
    region.imageExtent.depth = 1;

    commandBuffer->copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, { region });

    submitSingleTimeCommands(commandBuffer.get(), queue);
}

struct BufferObject {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory bufferMemory;
};

template <typename ElementType>
BufferObject constructDeviceLocalBuffer(vk::PhysicalDevice physicalDevice, vk::Device device,
    vk::CommandPool commandPool, vk::Queue graphicsQueue, vk::BufferUsageFlags usageFlags,
    std::vector<ElementType> const& bufferData)
{
    // create staging buffer
    const std::size_t bufferSize = sizeof(bufferData[0]) * bufferData.size();
    vk::UniqueBuffer stagingBuffer = makeBuffer(device, bufferSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(physicalDevice, device,
        stagingBuffer.get(), vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // fill staging buffer
    void* const data = device.mapMemory(stagingBufferMemory.get(), 0, bufferSize);
    std::memcpy(data, bufferData.data(), bufferSize);
    device.unmapMemory(stagingBufferMemory.get());

    // make vertex buffer
    BufferObject result;
    result.buffer = makeBuffer(device, bufferSize, vk::BufferUsageFlagBits::eTransferDst | usageFlags);
    result.bufferMemory = allocateBufferMemory(physicalDevice, device, result.buffer.get(),
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    // copy staging -> vertex buffer
    copyBuffer(stagingBuffer.get(), result.buffer.get(), bufferSize, device, commandPool, graphicsQueue);

    return result;
}

vk::UniqueDescriptorSetLayout makeDescriptorSetLayout(vk::Device device)
{
    std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{};

    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = vk::ShaderStageFlagBits::eVertex;
    layoutBindings[0].pImmutableSamplers = nullptr;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();

    return device.createDescriptorSetLayoutUnique(layoutInfo);
}

std::chrono::system_clock::time_point getCurrentTimePoint()
{
    return std::chrono::system_clock::now();
}

vk::UniqueDescriptorPool makeDescriptorPool(vk::Device device, std::uint32_t size)
{
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = size;
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = size;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = size;

    return device.createDescriptorPoolUnique(poolInfo);
}

std::vector<vk::UniqueDescriptorSet> makeDescriptorSets(vk::Device device, vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout, std::uint32_t size)
{
    std::vector<vk::DescriptorSetLayout> layouts(size, layout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();

    return device.allocateDescriptorSetsUnique(allocInfo);
}

void transitionImageLayout(vk::Device device, vk::CommandPool pool, vk::Queue queue, vk::Image image,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::UniqueCommandBuffer commandBuffer = beginSingleTimeCommands(device, pool);

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;

    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    } else {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
            | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    } else {
        throw std::invalid_argument("unsupported layout transition");
    }

    commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, { nullptr }, { nullptr }, { barrier });

    submitSingleTimeCommands(commandBuffer.get(), queue);
}

struct ImageObject {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory imageMemory;
    vk::Format format;
};

ImageObject makeImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::Extent2D extent, vk::Format format,
    vk::ImageUsageFlags usage)
{
    // create image objects
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.flags = {};

    vk::UniqueImage image = device.createImageUnique(imageInfo);

    // allocate memory to the image
    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::UniqueDeviceMemory imageMemory = device.allocateMemoryUnique(allocInfo);

    device.bindImageMemory(image.get(), imageMemory.get(), 0);

    return { std::move(image), std::move(imageMemory), imageInfo.format };
}

ImageObject makeTextureImage(vk::PhysicalDevice physicalDevice, vk::Device device, const char* filename,
    vk::CommandPool commandPool, vk::Queue queue)
{
    // read image file
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth * texHeight * 4);
    if (!pixels) {
        throw std::runtime_error("failed to load texture image");
    }

    vk::UniqueBuffer stagingBuffer = makeBuffer(device, imageSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(physicalDevice, device, stagingBuffer.get(),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // copy image data to buffer
    void* const data = device.mapMemory(stagingBufferMemory.get(), 0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    device.unmapMemory(stagingBufferMemory.get());

    stbi_image_free(pixels);

    // create image object
    ImageObject texture = makeImage(physicalDevice, device,
        { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight) },
        vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

    // image layout is not gpu accessible by now
    transitionImageLayout(device, commandPool, queue, texture.image.get(),
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    // copy buffer to image
    copyBufferToImage(stagingBuffer.get(), texture.image.get(),
        static_cast<std::uint32_t>(texWidth), static_cast<std::uint32_t>(texHeight),
        device, commandPool, queue);

    // change image layout
    transitionImageLayout(device, commandPool, queue, texture.image.get(),
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    return texture;
}

vk::UniqueSampler makeTextureSampler(vk::Device device)
{
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;

    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;

    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;

    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    return device.createSamplerUnique(samplerInfo);
}

ImageObject makeDepthImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::CommandPool commandPool,
    vk::Queue queue, vk::Extent2D extent)
{
    vk::Format depthFormat = [&]() {
        auto candidateFormats = { vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint };
        auto depthFormatIt = std::find_if(candidateFormats.begin(), candidateFormats.end(),
            [&](vk::Format format) {
                vk::FormatProperties props = physicalDevice.getFormatProperties(format);
                const auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
                return (props.optimalTilingFeatures & features) == features;
            });
        if (depthFormatIt == candidateFormats.end()) {
            throw std::runtime_error("no depth & stencil format supported");
        }

        return *depthFormatIt;
    }();

    // create image object
    ImageObject depth = makeImage(physicalDevice, device, extent, depthFormat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment);

    // make it a depth buffer
    transitionImageLayout(device, commandPool, queue, depth.image.get(),
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    return depth;
}

} // anonymous namespace

static const std::vector<Vertex> vertices = {
    { { -0.5f, -0.5f, 0.2f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.2f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.2f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.2f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f } },

    { { -0.5f, -0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f } },
};

static const std::vector<std::uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

VulkanApplication::VulkanApplication()
    : m_window(makeWindow(1024, 768))
    , m_instance(makeInstance())
    , m_dispatchLoader(m_instance.get())
    , m_callback(makeDebugMessenger(m_instance.get(), m_dispatchLoader))
    , m_surface(makeSurface(m_window.get(), m_instance.get()))
{
    const vk::PhysicalDevice physicalDevice = selectPhysicalDevice(m_instance.get());
    const QueueFamilyIndices queueFamilies = selectQueueFamilyIndices(physicalDevice, m_surface.get());
    m_device = makeDevice(queueFamilies, physicalDevice);

    // retrieve the graphics queue handles
    m_graphicsQueue = m_device->getQueue(queueFamilies.graphicsFamilyIndex, 0);
    m_presentQueue = m_device->getQueue(queueFamilies.presentFamilyIndex, 0);

    // make swapchain
    m_swapchainProps = selectSwapchainProperties(physicalDevice, m_surface.get(), m_window.get());
    m_swapchain = makeSwapchain(m_swapchainProps, queueFamilies, m_surface.get(), m_device.get());
    m_swapchainImages = retrieveSwapchainImages(m_device.get(), m_swapchain.get());
    const std::uint32_t swapchainImageCount = static_cast<std::uint32_t>(m_swapchainImages.size());
    for (const auto& image : m_swapchainImages) {
        m_swapchainImageViews.push_back(makeImageView(m_device.get(), image, m_swapchainProps.surfaceFormat.format));
    }

    // make command pool and command buffers
    m_commandPool = makeCommandPool(m_device.get(), queueFamilies.graphicsFamilyIndex);
    m_commandBuffers = allocateCommandBuffers(m_device.get(), m_commandPool.get(), swapchainImageCount);

    // make depth buffer
    ImageObject depth = makeDepthImage(physicalDevice, m_device.get(), m_commandPool.get(),
        m_graphicsQueue, m_swapchainProps.extent);
    m_depthImage = std::move(depth.image);
    m_depthImageMemory = std::move(depth.imageMemory);
    m_depthImageView = makeImageView(m_device.get(), m_depthImage.get(), depth.format, vk::ImageAspectFlagBits::eDepth);

    // make render pass
    m_renderPass = makeRenderPass(m_device.get(), m_swapchainProps.surfaceFormat.format, depth.format);

    // make description set layout, pool, and sets
    m_descriptorSetLayout = makeDescriptorSetLayout(m_device.get());
    m_descriptorPool = makeDescriptorPool(m_device.get(), swapchainImageCount);
    m_descriptorSets = makeDescriptorSets(m_device.get(), m_descriptorPool.get(),
        m_descriptorSetLayout.get(), swapchainImageCount);

    // make pipeline
    m_pipelineLayout = makePipelineLayout(m_device.get(), m_descriptorSetLayout.get());
    m_graphicsPipeline = makePipeline(m_device.get(), m_pipelineLayout.get(), m_swapchainProps.extent, m_renderPass.get());

    // make framebuffers
    m_framebuffers = makeFramebuffers(m_device.get(), m_swapchainImageViews, m_depthImageView.get(),
        m_renderPass.get(), m_swapchainProps.extent);

    // make semaphores for synchronizing frame drawing operations
    for (std::size_t index = 0; index < maxFramesInFlight; ++index) {
        m_imageAvailableSemaphores.push_back(makeSemaphore(m_device.get()));
        m_renderFinishedSemaphores.push_back(makeSemaphore(m_device.get()));
        m_inFlightFences.push_back(makeFence(m_device.get()));
    }

    // make & fill vertex buffer
    BufferObject vertexBuffer = constructDeviceLocalBuffer(physicalDevice, m_device.get(), m_commandPool.get(),
        m_graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertices);
    m_vertexBuffer = std::move(vertexBuffer.buffer);
    m_vertexBufferMemory = std::move(vertexBuffer.bufferMemory);

    // make & fill index buffer
    BufferObject indexBuffer = constructDeviceLocalBuffer(physicalDevice, m_device.get(), m_commandPool.get(),
        m_graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indices);
    m_indexBuffer = std::move(indexBuffer.buffer);
    m_indexBufferMemory = std::move(indexBuffer.bufferMemory);

    // make uniform buffers
    for (std::uint32_t i = 0; i < swapchainImageCount; ++i) {
        m_uniformBuffers.push_back(makeBuffer(m_device.get(), sizeof(UniformBufferObject),
            vk::BufferUsageFlagBits::eUniformBuffer));
        m_uniformBuffersMemory.push_back(allocateBufferMemory(physicalDevice, m_device.get(), m_uniformBuffers.back().get(),
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    }

    // make textures
    ImageObject texture = makeTextureImage(physicalDevice, m_device.get(), "textures/texture.jpg",
        m_commandPool.get(), m_graphicsQueue);
    m_textureImage = std::move(texture.image);
    m_textureImageMemory = std::move(texture.imageMemory);
    m_textureImageView = makeImageView(m_device.get(), m_textureImage.get(), texture.format);

    // make sampler
    m_sampler = makeTextureSampler(m_device.get());

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::run()
{
    m_startTime = m_lastFpsTimePoint = getCurrentTimePoint();
    m_fpsCounter = 0;

    // main loop
    while (!glfwWindowShouldClose(m_window.get())) {
        glfwPollEvents();

        drawFrame();

        // calculate FPS
        m_fpsCounter++;
        auto elapsedTime = getCurrentTimePoint() - m_lastFpsTimePoint;

        using namespace std::chrono;
        if (elapsedTime >= seconds(1)) {
            double fps = static_cast<double>(m_fpsCounter) / duration_cast<seconds>(elapsedTime).count();
            std::cout << "FPS: " << std::fixed << std::setprecision(0) << fps << std::endl;

            m_lastFpsTimePoint = getCurrentTimePoint();
            m_fpsCounter = 0;
        }
    }

    m_device->waitIdle();
}

void VulkanApplication::recordDrawCommands()
{
    std::size_t index = 0;
    for (vk::UniqueCommandBuffer const& commandBuffer : m_commandBuffers) {
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        beginInfo.pInheritanceInfo = nullptr;

        commandBuffer->begin(beginInfo);

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = m_renderPass.get();
        renderPassInfo.framebuffer = m_framebuffers[index].get();
        renderPassInfo.renderArea.offset.x = 0;
        renderPassInfo.renderArea.offset.y = 0;
        renderPassInfo.renderArea.extent = m_swapchainProps.extent;

        std::array<vk::ClearValue, 2> clearValues{};

        // color buffer clear value
        clearValues[0].color.float32[0] = 0.0f;
        clearValues[0].color.float32[1] = 0.0f;
        clearValues[0].color.float32[2] = 0.0f;
        clearValues[0].color.float32[3] = 1.0f;

        // depth buffer clear value
        clearValues[1].depthStencil.depth = 1.0f;
        clearValues[1].depthStencil.stencil = 0;

        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();

        commandBuffer->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline.get());

        commandBuffer->bindVertexBuffers(0, { m_vertexBuffer.get() }, { 0 });

        commandBuffer->bindIndexBuffer(m_indexBuffer.get(), 0, vk::IndexType::eUint16);

        // bind uniform descriptor sets
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_uniformBuffers[index].get();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = m_textureImageView.get();
        imageInfo.sampler = m_sampler.get();

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].dstSet = m_descriptorSets[index].get();
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = m_descriptorSets[index].get();
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        m_device->updateDescriptorSets(descriptorWrites, {});

        commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout.get(), 0,
            { m_descriptorSets[index].get() }, {});

        commandBuffer->drawIndexed(static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer->endRenderPass();

        commandBuffer->end();

        index++;
    }
}

void VulkanApplication::drawFrame()
{
    // The drawFrame function will perform the following operations:
    // 1. Acquire an image from the swap chain
    // 2. Execute the command buffer with that image as attachment in the framebuffer
    // 3. Return the image to the swap chain for presentation

    std::array<vk::Fence, 1> inFlightFences = { { m_inFlightFences[m_currentFrame].get() } };
    m_device->waitForFences(inFlightFences, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    m_device->resetFences(inFlightFences);

    // acquire next image to write into from the swap chain
    // note: this function is asynchronous
    std::uint32_t imageIndex;
    m_device->acquireNextImageKHR(m_swapchain.get(), std::numeric_limits<std::uint64_t>::max(),
        m_imageAvailableSemaphores[m_currentFrame].get(), nullptr, &imageIndex);

    // update the uniform data
    {
        using namespace std::chrono;
        float time = duration<float, seconds::period>(getCurrentTimePoint() - m_startTime).count();
        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(1.0f, 1.0f, ((glm::sin(time) + 1.0f) / 2.0f) * 2.0f),
            glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(m_swapchainProps.extent.width) / m_swapchainProps.extent.height,
            0.1f, 10.0f);
        ubo.proj[1][1] *= -1; // invert Y axis

        void* const data = m_device->mapMemory(m_uniformBuffersMemory[imageIndex].get(), 0, sizeof(ubo));
        std::memcpy(data, &ubo, sizeof(ubo));
        m_device->unmapMemory(m_uniformBuffersMemory[imageIndex].get());
    }

    // execute the command buffer
    vk::SubmitInfo submitInfo{};

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    std::array<vk::Semaphore, 1> imageAvailableSemaphores = { { m_imageAvailableSemaphores[m_currentFrame].get() } };
    submitInfo.waitSemaphoreCount = imageAvailableSemaphores.size();
    submitInfo.pWaitSemaphores = imageAvailableSemaphores.data();
    submitInfo.pWaitDstStageMask = &waitStage;

    std::array<vk::CommandBuffer, 1> commandBuffers = { { m_commandBuffers[imageIndex].get() } };
    submitInfo.commandBufferCount = commandBuffers.size();
    submitInfo.pCommandBuffers = commandBuffers.data();

    std::array<vk::Semaphore, 1> renderFinishedSemaphores = { { m_renderFinishedSemaphores[m_currentFrame].get() } };
    submitInfo.signalSemaphoreCount = renderFinishedSemaphores.size();
    submitInfo.pSignalSemaphores = renderFinishedSemaphores.data();

    m_graphicsQueue.submit(1, &submitInfo, m_inFlightFences[m_currentFrame].get());

    // write back to the swap chain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = renderFinishedSemaphores.size();
    presentInfo.pWaitSemaphores = renderFinishedSemaphores.data();

    std::array<vk::SwapchainKHR, 1> swapchains = { { m_swapchain.get() } };
    presentInfo.swapchainCount = swapchains.size();
    presentInfo.pSwapchains = swapchains.data();
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr;

    m_presentQueue.presentKHR(presentInfo);

    m_currentFrame = (m_currentFrame + 1) % maxFramesInFlight;
}

int main()
{
    try {
        VulkanApplication app;
        app.run();
    } catch (std::exception const& err) {
        std::cerr << "Error while running the program: " << err.what() << std::endl;
    }

    return 0;
}
