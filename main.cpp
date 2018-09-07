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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

        float queuePriorities[] = { 1.0f };
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = queuePriorities;

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

        std::uint32_t queueFamilyIndices[2] = { queueFamilies.graphicsFamilyIndex, queueFamilies.presentFamilyIndex };
        swapChainCreateInfo.queueFamilyIndexCount = 2;
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
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

std::vector<vk::UniqueImageView> makeSwapchainImageViews(
    vk::Device device, std::vector<vk::Image> const& images, vk::Format imageFormat)
{
    std::vector<vk::UniqueImageView> imageViews;
    for (const auto& image : images) {
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
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        imageViews.push_back(device.createImageViewUnique(imageViewCreateInfo));
    }

    return imageViews;
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

vk::UniqueRenderPass makeRenderPass(vk::Device device, vk::Format imageFormat)
{
    // make render pass
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = imageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRefs[1]{};
    // layout(location = 0) out vec4 outColor
    colorAttachmentRefs[0].attachment = 0;
    colorAttachmentRefs[0].layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = colorAttachmentRefs;

    // wait until the image is loaded
    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = {};
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
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
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

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

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule.get();
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule.get();
    fragShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

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

    vk::DynamicState dynamicStates[] = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eLineWidth
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
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
    std::vector<vk::UniqueImageView> const& imageViews, vk::RenderPass renderPass, vk::Extent2D swapChainExtent)
{
    std::vector<vk::UniqueFramebuffer> framebuffers;

    for (auto const& uniqueImageView : imageViews) {
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderPass;

        // number of attached ImageViews
        framebufferInfo.attachmentCount = 1;

        vk::ImageView imageView = uniqueImageView.get();
        framebufferInfo.pAttachments = &imageView;

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

void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size,
    vk::Device device, vk::CommandPool pool, vk::Queue queue)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;

    vk::UniqueCommandBuffer commandBuffer = std::move(device.allocateCommandBuffersUnique(allocInfo)[0]);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer->begin(beginInfo);

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer->copyBuffer(srcBuffer, dstBuffer, { copyRegion });

    commandBuffer->end();

    vk::SubmitInfo submitInfo{};
    vk::CommandBuffer commandBuffers[] = { commandBuffer.get() };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffers;

    queue.submit({ submitInfo }, nullptr);
    queue.waitIdle();
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

vk::UniqueDescriptorSetLayout makeUniformDescriptorSetLayout(vk::Device device, vk::ShaderStageFlags shaderStage)
{
    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = shaderStage;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    return device.createDescriptorSetLayoutUnique(layoutInfo);
}

std::chrono::system_clock::time_point getCurrentTimePoint()
{
    return std::chrono::system_clock::now();
}

vk::UniqueDescriptorPool makeDescriptorPool(vk::Device device, std::uint32_t size)
{
    vk::DescriptorPoolSize poolSize{};
    poolSize.type = vk::DescriptorType::eUniformBuffer;
    poolSize.descriptorCount = size;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
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

} // anonymous namespace

static const std::vector<Vertex> vertices = {
    { { -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
    { { 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } },
    { { -0.5f, 0.5f }, { 1.0f, 1.0f, 1.0f } },
};

static const std::vector<std::uint16_t> indices = { 0, 1, 2, 2, 3, 0 };

VulkanApplication::VulkanApplication()
    : m_window(makeWindow(800, 600))
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
    m_swapchainImageViews = makeSwapchainImageViews(m_device.get(), m_swapchainImages, m_swapchainProps.surfaceFormat.format);

    // make render pass
    m_renderPass = makeRenderPass(m_device.get(), m_swapchainProps.surfaceFormat.format);

    // make description set layout
    m_descriptorSetLayout = makeUniformDescriptorSetLayout(m_device.get(), vk::ShaderStageFlagBits::eVertex);

    // make pipeline
    m_pipelineLayout = makePipelineLayout(m_device.get(), m_descriptorSetLayout.get());
    m_graphicsPipeline = makePipeline(m_device.get(), m_pipelineLayout.get(), m_swapchainProps.extent, m_renderPass.get());

    // make framebuffers
    m_framebuffers = makeFramebuffers(m_device.get(), m_swapchainImageViews, m_renderPass.get(), m_swapchainProps.extent);

    // make command pool and command buffers
    m_commandPool = makeCommandPool(m_device.get(), queueFamilies.graphicsFamilyIndex);
    m_commandBuffers = allocateCommandBuffers(m_device.get(), m_commandPool.get(), swapchainImageCount);

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

    // make descriptor pool
    m_descriptorPool = makeDescriptorPool(m_device.get(), swapchainImageCount);

    // make descriptor sets
    m_descriptorSets = makeDescriptorSets(m_device.get(), m_descriptorPool.get(),
        m_descriptorSetLayout.get(), swapchainImageCount);

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

        vk::ClearValue clearColor{};
        clearColor.color.float32[0] = 0.0f;
        clearColor.color.float32[1] = 0.0f;
        clearColor.color.float32[2] = 0.0f;
        clearColor.color.float32[3] = 1.0f;

        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        commandBuffer->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline.get());

        commandBuffer->bindVertexBuffers(0, { m_vertexBuffer.get() }, { 0 });

        commandBuffer->bindIndexBuffer(m_indexBuffer.get(), 0, vk::IndexType::eUint16);

        // bind uniform descriptor sets
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_uniformBuffers[index].get();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = m_descriptorSets[index].get();
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;

        descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;

        m_device->updateDescriptorSets({ descriptorWrite }, {});

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

    vk::Fence inFlightFences[] = { m_inFlightFences[m_currentFrame].get() };
    m_device->waitForFences(1, inFlightFences, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    m_device->resetFences(1, inFlightFences);

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
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
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
    vk::Semaphore imageAvailableSemaphores[] = { m_imageAvailableSemaphores[m_currentFrame].get() };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = imageAvailableSemaphores;
    submitInfo.pWaitDstStageMask = &waitStage;

    vk::CommandBuffer commandBuffers[] = { m_commandBuffers[imageIndex].get() };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffers;

    vk::Semaphore renderFinishedSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame].get() };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = renderFinishedSemaphores;

    m_graphicsQueue.submit(1, &submitInfo, m_inFlightFences[m_currentFrame].get());

    // write back to the swap chain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = renderFinishedSemaphores;

    vk::SwapchainKHR swapchains[] = { m_swapchain.get() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
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
