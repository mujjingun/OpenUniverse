#include "vulkan_routines.h"

#include <bitset>
#include <fstream>
#include <iostream>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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

std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)> ou::makeWindow(int width, int height)
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

vk::UniqueInstance ou::makeInstance()
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

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
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

vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> ou::makeDebugMessenger(
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

vk::UniqueSurfaceKHR ou::makeSurface(GLFWwindow* window, vk::Instance instance)
{
    VkSurfaceKHR surface;

    // create window surface
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface");
    }

    vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic> surfaceDeleter(instance);
    return vk::UniqueSurfaceKHR(surface, surfaceDeleter);
}

vk::PhysicalDevice ou::selectPhysicalDevice(vk::Instance instance)
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

ou::QueueFamilyIndices ou::selectQueueFamilyIndices(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
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

vk::UniqueDevice ou::makeDevice(QueueFamilyIndices queueFamilies, vk::PhysicalDevice physicalDevice)
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

ou::SwapchainProperties ou::selectSwapchainProperties(
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

vk::UniqueSwapchainKHR ou::makeSwapchain(
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

std::vector<vk::Image> ou::retrieveSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain)
{
    // retrieve swap chain image handles
    return device.getSwapchainImagesKHR(swapchain);
}

vk::UniqueImageView ou::makeImageView(vk::Device device, vk::Image image, vk::Format imageFormat,
    vk::ImageAspectFlags imageType, uint32_t mipLevels)
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

    // mipmapping
    imageViewCreateInfo.subresourceRange.aspectMask = imageType;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = mipLevels;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    return device.createImageViewUnique(imageViewCreateInfo);
}

static std::vector<char> readFile(const char* fileName)
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

static vk::UniqueShaderModule createShaderModule(vk::Device device, std::vector<char> const& code)
{
    vk::ShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.codeSize = code.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

    return device.createShaderModuleUnique(shaderModuleCreateInfo);
}

vk::UniqueRenderPass ou::makeRenderPass(vk::Device device, vk::SampleCountFlagBits sampleCount,
    vk::Format imageFormat, vk::Format depthFormat)
{
    std::array<vk::AttachmentDescription, 3> attachments;

    // color attachment
    attachments[0].format = imageFormat;
    attachments[0].samples = sampleCount;
    attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[0].initialLayout = vk::ImageLayout::eUndefined;
    attachments[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

    // depth attachment
    attachments[1].format = depthFormat;
    attachments[1].samples = sampleCount;
    attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].initialLayout = vk::ImageLayout::eUndefined;
    attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    // multisample resolve attachment
    attachments[2].format = imageFormat;
    attachments[2].samples = vk::SampleCountFlagBits::e1;
    attachments[2].loadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[2].storeOp = vk::AttachmentStoreOp::eStore;
    attachments[2].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachments[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[2].initialLayout = vk::ImageLayout::eUndefined;
    attachments[2].finalLayout = vk::ImageLayout::ePresentSrcKHR;

    std::array<vk::AttachmentReference, 1> colorAttachmentRefs{};
    // layout(location = 0) out vec4 outColor
    colorAttachmentRefs[0].attachment = 0;
    colorAttachmentRefs[0].layout = vk::ImageLayout::eColorAttachmentOptimal;

    // only 1 depth attachment allowed
    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference multiSampleResolveRef{};
    multiSampleResolveRef.attachment = 2;
    multiSampleResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = colorAttachmentRefs.size();
    subpass.pColorAttachments = colorAttachmentRefs.data();
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &multiSampleResolveRef;

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

vk::UniquePipelineLayout ou::makePipelineLayout(vk::Device device, vk::DescriptorSetLayout descriptorSetLayout)
{
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    return device.createPipelineLayoutUnique(pipelineLayoutInfo);
}

vk::UniquePipeline ou::makePipeline(
    vk::Device device, vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent, vk::RenderPass renderPass,
    vk::SampleCountFlagBits sampleCount, vk::VertexInputBindingDescription bindingDescription,
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions)
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
    multisampling.rasterizationSamples = sampleCount;
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

std::vector<vk::UniqueFramebuffer> ou::makeFramebuffers(vk::Device device,
    std::vector<vk::UniqueImageView> const& imageViews, vk::ImageView depthImageView, vk::ImageView multiSampleImageView,
    vk::RenderPass renderPass, vk::Extent2D swapChainExtent)
{
    std::vector<vk::UniqueFramebuffer> framebuffers;

    for (auto const& uniqueImageView : imageViews) {
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderPass;

        std::array<vk::ImageView, 3> attachments = {
            { multiSampleImageView, depthImageView, uniqueImageView.get() }
        };
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();

        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        framebuffers.push_back(device.createFramebufferUnique(framebufferInfo));
    }

    return framebuffers;
}

vk::UniqueCommandPool ou::makeCommandPool(vk::Device device, std::uint32_t queueFamilyIndex)
{
    vk::CommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    poolCreateInfo.flags = {};

    return device.createCommandPoolUnique(poolCreateInfo);
}

std::vector<vk::UniqueCommandBuffer> ou::allocateCommandBuffers(vk::Device device, vk::CommandPool commandPool,
    std::uint32_t count)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = count;

    return device.allocateCommandBuffersUnique(allocInfo);
}

vk::UniqueSemaphore ou::makeSemaphore(vk::Device device)
{
    vk::SemaphoreCreateInfo semaphoreInfo{};
    return device.createSemaphoreUnique(semaphoreInfo);
}

vk::UniqueFence ou::makeFence(vk::Device device)
{
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    return device.createFenceUnique(fenceInfo);
}

static std::uint32_t selectMemoryType(vk::PhysicalDevice physicalDevice,
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

vk::UniqueBuffer ou::makeBuffer(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::SharingMode sharingMode)
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = sharingMode;

    return device.createBufferUnique(bufferInfo);
}

vk::UniqueDeviceMemory ou::allocateBufferMemory(vk::PhysicalDevice physicalDevice, vk::Device device, vk::Buffer buffer,
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

static vk::UniqueCommandBuffer beginSingleTimeCommands(vk::Device device, vk::CommandPool pool)
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

static void submitSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::Queue queue)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    queue.submit({ submitInfo }, nullptr);
    queue.waitIdle();
}

static void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size,
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

static void copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, std::uint32_t width, std::uint32_t height,
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

ou::BufferObject ou::constructDeviceLocalBuffer(vk::PhysicalDevice physicalDevice, vk::Device device,
    vk::CommandPool commandPool, vk::Queue graphicsQueue, vk::BufferUsageFlags usageFlags,
    const void* bufferData, std::size_t bufferSize)
{
    // create staging buffer
    vk::UniqueBuffer stagingBuffer = makeBuffer(device, bufferSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(physicalDevice, device,
        stagingBuffer.get(), vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // fill staging buffer
    void* const data = device.mapMemory(stagingBufferMemory.get(), 0, bufferSize);
    std::memcpy(data, bufferData, bufferSize);
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

vk::UniqueDescriptorSetLayout ou::makeDescriptorSetLayout(vk::Device device)
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

std::chrono::system_clock::time_point ou::getCurrentTimePoint()
{
    return std::chrono::system_clock::now();
}

vk::UniqueDescriptorPool ou::makeDescriptorPool(vk::Device device, std::uint32_t size)
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

std::vector<vk::UniqueDescriptorSet> ou::makeDescriptorSets(vk::Device device, vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout, std::uint32_t size)
{
    std::vector<vk::DescriptorSetLayout> layouts(size, layout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();

    return device.allocateDescriptorSetsUnique(allocInfo);
}

static void transitionImageLayout(vk::Device device, vk::CommandPool pool, vk::Queue queue, vk::Image image,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout, std::uint32_t mipLevels)
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
    barrier.subresourceRange.levelCount = mipLevels;
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
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else {
        throw std::invalid_argument("unsupported layout transition");
    }

    commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, { nullptr }, { nullptr }, { barrier });

    submitSingleTimeCommands(commandBuffer.get(), queue);
}

ou::ImageObject ou::makeImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::SampleCountFlagBits numSamples,
    std::uint32_t mipLevels, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
{
    // create image objects
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.samples = numSamples;
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

    // make image view
    vk::UniqueImageView imageView = makeImageView(device, image.get(), imageInfo.format,
        vk::ImageAspectFlagBits::eColor, mipLevels);

    return {
        std::move(image),
        std::move(imageMemory),
        std::move(imageView),
        imageInfo.format
    };
}

static void generateMipmaps(vk::PhysicalDevice physicalDevice, vk::Device device, vk::CommandPool commandPool, vk::Queue queue,
    vk::Image image, vk::Format format, vk::Extent2D extent, std::uint32_t mipLevels)
{
    // check if image format can generate linear filter mipmaps
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(format);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting");
    }

    vk::UniqueCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    vk::ImageMemoryBarrier barrier{};
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    vk::Extent2D mipExtent = extent;
    for (std::uint32_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, { barrier });

        vk::ImageBlit blit{};
        blit.srcOffsets[0].x = 0;
        blit.srcOffsets[0].y = 0;
        blit.srcOffsets[0].z = 0;
        blit.srcOffsets[1].x = static_cast<std::int32_t>(mipExtent.width);
        blit.srcOffsets[1].y = static_cast<std::int32_t>(mipExtent.height);
        blit.srcOffsets[1].z = 1;
        blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;

        mipExtent.width = std::max(1u, mipExtent.width / 2);
        mipExtent.height = std::max(1u, mipExtent.height / 2);

        blit.dstOffsets[0].x = 0;
        blit.dstOffsets[0].y = 0;
        blit.dstOffsets[0].z = 0;
        blit.dstOffsets[1].x = static_cast<std::int32_t>(mipExtent.width);
        blit.dstOffsets[1].y = static_cast<std::int32_t>(mipExtent.height);
        blit.dstOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
            { blit }, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {}, { barrier });
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });

    submitSingleTimeCommands(commandBuffer.get(), queue);
}

ou::ImageObject ou::makeTextureImage(vk::PhysicalDevice physicalDevice, vk::Device device, const char* filename,
    vk::CommandPool commandPool, vk::Queue queue)
{
    // read image file
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("failed to load texture image");
    }

    const vk::Extent2D imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight) };
    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth * texHeight * 4);
    const std::uint32_t mipLevels = static_cast<std::uint32_t>(std::log2(std::max(texWidth, texHeight))) + 1;

    vk::UniqueBuffer stagingBuffer = makeBuffer(device, imageSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(physicalDevice, device, stagingBuffer.get(),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // copy image data to buffer
    void* const data = device.mapMemory(stagingBufferMemory.get(), 0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    device.unmapMemory(stagingBufferMemory.get());

    stbi_image_free(pixels);

    // create image object
    const vk::Format imageFormat = vk::Format::eR8G8B8A8Unorm;
    ImageObject texture = makeImage(physicalDevice, device, vk::SampleCountFlagBits::e1, mipLevels, imageExtent, imageFormat,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

    // image layout is not gpu accessible by now
    transitionImageLayout(device, commandPool, queue, texture.image.get(),
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);

    // copy buffer to image
    copyBufferToImage(stagingBuffer.get(), texture.image.get(),
        imageExtent.width, imageExtent.height, device, commandPool, queue);

    // generate mipmaps
    generateMipmaps(physicalDevice, device, commandPool, queue, texture.image.get(), imageFormat, imageExtent, mipLevels);

    return texture;
}

vk::UniqueSampler ou::makeTextureSampler(vk::Device device)
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
    samplerInfo.maxLod = 12.0f;

    return device.createSamplerUnique(samplerInfo);
}

vk::SampleCountFlagBits ou::getMaxUsableSampleCount(vk::PhysicalDevice physicalDevice)
{
    vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts = static_cast<vk::SampleCountFlags>(std::min(
        static_cast<std::uint32_t>(physicalDeviceProperties.limits.framebufferColorSampleCounts),
        static_cast<std::uint32_t>(physicalDeviceProperties.limits.framebufferDepthSampleCounts)));

    auto candidateSampleCounts = {
        vk::SampleCountFlagBits::e64, vk::SampleCountFlagBits::e32,
        vk::SampleCountFlagBits::e16, vk::SampleCountFlagBits::e8,
        vk::SampleCountFlagBits::e4, vk::SampleCountFlagBits::e2
    };

    for (auto candidate : candidateSampleCounts) {
        if (counts & candidate) {
            return candidate;
        }
    }
    return vk::SampleCountFlagBits::e1;
}

ou::ImageObject ou::makeDepthImage(vk::PhysicalDevice physicalDevice, vk::Device device, vk::CommandPool commandPool,
    vk::Queue queue, vk::Extent2D extent, vk::SampleCountFlagBits sampleCount)
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
    ImageObject depth = makeImage(physicalDevice, device, sampleCount, 1, extent, depthFormat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment);

    // make it a depth buffer
    transitionImageLayout(device, commandPool, queue, depth.image.get(),
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    return depth;
}

ou::ImageObject ou::makeMultiSampleImage(vk::PhysicalDevice physicalDevice, vk::Device device, VkCommandPool commandPool,
    vk::Queue queue, vk::Format imageFormat, vk::Extent2D extent, vk::SampleCountFlagBits sampleCount)
{
    ImageObject image = makeImage(physicalDevice, device, sampleCount, 1,
        extent, imageFormat, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment);

    transitionImageLayout(device, commandPool, queue, image.image.get(),
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1);

    return image;
}
