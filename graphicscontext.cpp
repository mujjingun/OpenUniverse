#include "graphicscontext.h"

#include <bitset>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

static const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

static const bool enableValidationLayers =
#ifdef NDEBUG
    false;
#else
    true;
#endif

ou::UniqueWindow ou::makeWindow(int width, int height, bool fullscreen)
{
    // initialize glfw context
    glfwInit();

    // tell glfw to not use OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // disable window resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    // create window
    GLFWwindow* window;
    if (fullscreen) {
        window = glfwCreateWindow(mode->width, mode->height, "OpenUniverse on Vulkan",
            glfwGetPrimaryMonitor(), nullptr);
    } else {
        window = glfwCreateWindow(width, height, "OpenUniverse on Vulkan", nullptr, nullptr);
    }

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

ou::UniqueDebugMessenger ou::makeDebugMessenger(
    vk::Instance instance, vk::DispatchLoaderDynamic const& dispatchLoader)
{
    // register debug callback
    if (enableValidationLayers) {
        vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose;
        //| vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
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
                && deviceFeatures.samplerAnisotropy;

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

    indices.graphics = static_cast<std::uint32_t>(graphicsFamilyIt - queueFamilies.begin());

    auto presentFamilyIt = std::find_if(queueFamilies.begin(), queueFamilies.end(),
        [&](vk::QueueFamilyProperties const& queueFamily) -> bool {
            const std::uint32_t index = static_cast<std::uint32_t>(&queueFamily - queueFamilies.data());
            vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(index, surface);
            return queueFamily.queueCount > 0 && presentSupport;
        });

    if (presentFamilyIt == queueFamilies.end()) {
        throw std::runtime_error("no presentation queue family found");
    }

    indices.presentation = static_cast<std::uint32_t>(presentFamilyIt - queueFamilies.begin());

    auto computeFamilyIt = std::find_if(queueFamilies.begin(), queueFamilies.end(),
        [&](vk::QueueFamilyProperties const& queueFamily) -> bool {
            return queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eCompute;
        });

    if (computeFamilyIt == queueFamilies.end()) {
        throw std::runtime_error("no compute queue family found");
    }

    indices.compute = static_cast<std::uint32_t>(computeFamilyIt - queueFamilies.begin());

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
    static const std::array<const char*, 1> requiredDeviceExtensions = {
        { VK_KHR_SWAPCHAIN_EXTENSION_NAME }
    };
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
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    vk::DeviceQueueCreateInfo graphicQueueInfo{};
    graphicQueueInfo.queueFamilyIndex = queueFamilies.graphics;
    std::vector<float> graphicsPriorities = { 1, 0 };
    graphicQueueInfo.queueCount = static_cast<std::uint32_t>(graphicsPriorities.size());
    graphicQueueInfo.pQueuePriorities = graphicsPriorities.data();
    queueCreateInfos.push_back(graphicQueueInfo);

    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.tessellationShader = VK_TRUE;
    deviceFeatures.shaderTessellationAndGeometryPointSize = VK_TRUE;

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

vk::UniqueCommandPool ou::makeCommandPool(vk::Device device, std::uint32_t queueFamilyIndex)
{
    vk::CommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    poolCreateInfo.flags = {};

    return device.createCommandPoolUnique(poolCreateInfo);
}

ou::GraphicsContext::GraphicsContext()
    : GraphicsContext(600, 480, false)
{
}

ou::GraphicsContext::GraphicsContext(int width, int height, bool fullscreen)
    : m_window(makeWindow(width, height, fullscreen))
    , m_videoMode(glfwGetVideoMode(glfwGetPrimaryMonitor()))
    , m_windowedModeSize{ static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height) }
    , m_instance(makeInstance())
    , m_dispatchLoader(*m_instance)
    , m_messenger(makeDebugMessenger(*m_instance, m_dispatchLoader))
    , m_surface(makeSurface(m_window.get(), *m_instance))

    // find suitable physical device
    , m_physicalDevice(selectPhysicalDevice(*m_instance))

    // find graphics & presentation queue indices
    , m_queueIndices(selectQueueFamilyIndices(m_physicalDevice, *m_surface))

    // make logical device
    , m_device(makeDevice(m_queueIndices, m_physicalDevice))

    // retrieve the graphics queue handles
    , m_graphicsQueue(m_device->getQueue(m_queueIndices.graphics, 0))
    , m_computeQueue(m_device->getQueue(m_queueIndices.compute, 1))
    , m_presentQueue(m_device->getQueue(m_queueIndices.presentation, 0))

    // make command pool
    , m_commandPool(makeCommandPool(*m_device, m_queueIndices.graphics))
{
}

vk::Device ou::GraphicsContext::device() const
{
    return *m_device;
}

GLFWwindow* ou::GraphicsContext::window() const
{
    return m_window.get();
}

vk::Queue ou::GraphicsContext::graphicsQueue() const
{
    return m_graphicsQueue;
}

vk::Queue ou::GraphicsContext::computeQueue() const
{
    return m_computeQueue;
}

vk::Queue ou::GraphicsContext::presentQueue() const
{
    return m_presentQueue;
}

int ou::GraphicsContext::refreshRate() const
{
    return m_videoMode->refreshRate;
}

vk::Extent2D ou::GraphicsContext::screenResolution() const
{
    return { static_cast<std::uint32_t>(m_videoMode->width),
        static_cast<std::uint32_t>(m_videoMode->height) };
}

bool ou::GraphicsContext::isFullscreen() const
{
    GLFWmonitor* monitor = glfwGetWindowMonitor(m_window.get());
    return monitor != nullptr;
}

void ou::GraphicsContext::toggleFullscreenMode() const
{
    if (!isFullscreen()) {
        // enable fullscreen mode
        glfwSetWindowMonitor(m_window.get(), glfwGetPrimaryMonitor(), 0, 0,
            static_cast<int>(screenResolution().width), static_cast<int>(screenResolution().height), refreshRate());
    } else {
        // disable fullscreen mode
        glfwSetWindowMonitor(m_window.get(), nullptr, 0, 0,
            static_cast<int>(m_windowedModeSize.width), static_cast<int>(m_windowedModeSize.height), refreshRate());
    }
}

ou::SwapchainProperties ou::GraphicsContext::selectSwapchainProperties() const
{
    // set up the swap chain
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);

    auto surfaceFormats = m_physicalDevice.getSurfaceFormatsKHR(*m_surface);
    auto surfacePresentModes = m_physicalDevice.getSurfacePresentModesKHR(*m_surface);

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
            }
            if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = vk::PresentModeKHR::eImmediate;
            }
        }

        return bestMode;
    }();

    std::cout << "Present mode: ";
    switch (properties.presentMode) {
    case vk::PresentModeKHR::eFifo:
        std::cout << "FIFO";
        break;
    case vk::PresentModeKHR::eFifoRelaxed:
        std::cout << "FIFO_RELAXED";
        break;
    case vk::PresentModeKHR::eImmediate:
        std::cout << "IMMEDIATE";
        break;
    case vk::PresentModeKHR::eMailbox:
        std::cout << "MAILBOX";
        break;
    case vk::PresentModeKHR::eSharedContinuousRefresh:
        std::cout << "SHARED_CONTINUOUS_REFRESH";
        break;
    case vk::PresentModeKHR::eSharedDemandRefresh:
        std::cout << "SHARED_DEMAND_REFRESH";
        break;
    }
    std::cout << std::endl;

    // choose swap surface dimensions
    properties.extent = [&]() {
        if (surfaceCapabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return surfaceCapabilities.currentExtent;
        } else {
            int windowWidth, windowHeight;
            glfwGetFramebufferSize(m_window.get(), &windowWidth, &windowHeight);

            vk::Extent2D actualExtent{};
            actualExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
                std::min(surfaceCapabilities.maxImageExtent.width, static_cast<std::uint32_t>(windowWidth)));
            actualExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
                std::min(surfaceCapabilities.maxImageExtent.height, static_cast<std::uint32_t>(windowHeight)));

            return actualExtent;
        }
    }();

    // choose number of images to use
    properties.imageCount = [&]() {
        std::uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
            imageCount = surfaceCapabilities.maxImageCount;
        }
        return imageCount;
    }();

    properties.transform = surfaceCapabilities.currentTransform;

    return properties;
}

vk::UniqueDescriptorSetLayout ou::GraphicsContext::makeDescriptorSetLayout(std::vector<vk::DescriptorType> const& types,
    std::vector<vk::ShaderStageFlags> const& stages, std::vector<std::uint32_t> const& counts) const
{
    assert(types.size() == stages.size() && types.size() == counts.size());
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings(types.size());

    for (std::size_t index = 0; index < layoutBindings.size(); ++index) {
        layoutBindings[index].binding = static_cast<std::uint32_t>(index);
        layoutBindings[index].descriptorType = types[index];
        layoutBindings[index].descriptorCount = counts[index];
        layoutBindings[index].stageFlags = stages[index];
        layoutBindings[index].pImmutableSamplers = nullptr;
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    return m_device->createDescriptorSetLayoutUnique(layoutInfo);
}

vk::UniqueDescriptorPool ou::GraphicsContext::makeDescriptorPool(uint32_t size, std::vector<vk::DescriptorType> const& types) const
{
    std::vector<vk::DescriptorPoolSize> poolSizes(types.size());
    for (std::size_t index = 0; index < types.size(); ++index) {
        poolSizes[index].type = types[index];
        poolSizes[index].descriptorCount = size;
    }

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = size;

    return m_device->createDescriptorPoolUnique(poolInfo);
}

std::vector<vk::DescriptorSet> ou::GraphicsContext::makeDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
    uint32_t size) const
{
    std::vector<vk::DescriptorSetLayout> layouts(size, layout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();

    return m_device->allocateDescriptorSets(allocInfo);
}

ou::DescriptorSetObject ou::GraphicsContext::makeDescriptorSet(uint32_t size, std::vector<vk::DescriptorType> const& types,
    const std::vector<vk::ShaderStageFlags>& stages, std::vector<std::uint32_t> const& counts) const
{
    DescriptorSetObject set;
    set.layout = makeDescriptorSetLayout(types, stages, counts);
    set.pool = makeDescriptorPool(size, types);
    set.sets = makeDescriptorSets(*set.pool, *set.layout, size);

    return set;
}

vk::UniqueSwapchainKHR ou::GraphicsContext::makeSwapchain(ou::SwapchainProperties props, vk::SwapchainKHR oldSwapchain) const
{
    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.surface = *m_surface;
    swapChainCreateInfo.minImageCount = props.imageCount;
    swapChainCreateInfo.imageFormat = props.surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = props.surfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = props.extent;
    swapChainCreateInfo.presentMode = props.presentMode;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    if (m_queueIndices.graphics != m_queueIndices.presentation) {
        // image needs to shared by two queues
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;

        std::array<std::uint32_t, 2> queueFamilyIndices = {
            { m_queueIndices.graphics, m_queueIndices.presentation }
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
    swapChainCreateInfo.oldSwapchain = oldSwapchain;

    return m_device->createSwapchainKHRUnique(swapChainCreateInfo);
}

std::vector<vk::Image> ou::GraphicsContext::retrieveSwapchainImages(vk::SwapchainKHR swapchain) const
{
    // retrieve swap chain image handles
    return m_device->getSwapchainImagesKHR(swapchain);
}

vk::UniqueImageView ou::GraphicsContext::makeImageView(vk::Image image, vk::Format imageFormat,
    vk::ImageAspectFlags imageType, uint32_t mipLevels, vk::ImageViewType dimensions, std::uint32_t layerCount) const
{
    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = dimensions;
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
    imageViewCreateInfo.subresourceRange.layerCount = layerCount;

    return m_device->createImageViewUnique(imageViewCreateInfo);
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

ou::ImageObject ou::GraphicsContext::makeImage(vk::SampleCountFlagBits numSamples, uint32_t mipLevels,
    vk::Extent2D extent, std::uint32_t layerCount,
    vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlagBits aspect) const
{
    // create image objects
    vk::ImageCreateInfo imageInfo{};

    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = static_cast<std::uint32_t>(layerCount);
    imageInfo.format = format;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.samples = numSamples;
    imageInfo.flags = {};

    vk::UniqueImage image = m_device->createImageUnique(imageInfo);

    // allocate memory to the image
    vk::MemoryRequirements memRequirements = m_device->getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::UniqueDeviceMemory imageMemory = m_device->allocateMemoryUnique(allocInfo);

    m_device->bindImageMemory(image.get(), imageMemory.get(), 0);

    // make image view
    vk::UniqueImageView imageView = makeImageView(image.get(), imageInfo.format, aspect,
        mipLevels, layerCount > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D, layerCount);

    return {
        std::move(image),
        std::move(imageMemory),
        std::move(imageView),
        imageInfo.format,
        extent,
        mipLevels,
        layerCount
    };
}

ou::SingleTimeCommandBuffer ou::GraphicsContext::beginSingleTimeCommands() const
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = *m_commandPool;
    allocInfo.commandBufferCount = 1;

    vk::UniqueCommandBuffer commandBuffer = std::move(m_device->allocateCommandBuffersUnique(allocInfo)[0]);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer->begin(beginInfo);

    return SingleTimeCommandBuffer(std::move(commandBuffer), m_graphicsQueue);
}

void ou::transitionImageLayout(vk::CommandBuffer commandBuf, vk::Image image, std::uint32_t layerCount,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout, std::uint32_t mipLevels, std::uint32_t baseMipLevel)
{
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

    barrier.subresourceRange.baseMipLevel = baseMipLevel;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layerCount;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eGeneral) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = {};

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eAllCommands;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
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
        destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else {
        throw std::invalid_argument("unsupported layout transition");
    }

    commandBuf.pipelineBarrier(sourceStage, destinationStage, {}, { nullptr }, { nullptr }, { barrier });
}

ou::ImageObject ou::GraphicsContext::makeDepthImage(vk::Extent2D extent, vk::SampleCountFlagBits sampleCount, vk::ImageUsageFlags extraFlags) const
{
    vk::Format depthFormat = [&]() {
        auto candidateFormats = { vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint };
        auto depthFormatIt = std::find_if(candidateFormats.begin(), candidateFormats.end(),
            [&](vk::Format format) {
                vk::FormatProperties props = m_physicalDevice.getFormatProperties(format);
                const auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
                return (props.optimalTilingFeatures & features) == features;
            });
        if (depthFormatIt == candidateFormats.end()) {
            throw std::runtime_error("no depth & stencil format supported");
        }

        return *depthFormatIt;
    }();

    // create image object
    ImageObject depth = makeImage(sampleCount, 1, extent, 1, depthFormat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | extraFlags, vk::ImageAspectFlagBits::eDepth);

    // make it a depth buffer
    transitionImageLayout(*beginSingleTimeCommands(), depth.image.get(), 1,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    return depth;
}

ou::ImageObject ou::GraphicsContext::makeMultiSampleImage(vk::Format imageFormat, vk::Extent2D extent, std::uint32_t layerCount,
    vk::SampleCountFlagBits sampleCount) const
{
    ImageObject image = makeImage(sampleCount, 1, extent, layerCount, imageFormat,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::ImageAspectFlagBits::eColor);

    transitionImageLayout(*beginSingleTimeCommands(), image.image.get(), layerCount,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1);

    return image;
}

vk::UniqueRenderPass ou::GraphicsContext::makeRenderPass(std::vector<vk::SubpassDescription> const& subpasses,
    std::vector<vk::SubpassDependency> const& dependencies, std::vector<vk::AttachmentDescription> const& attachments) const
{
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = static_cast<std::uint32_t>(subpasses.size());
    renderPassInfo.pSubpasses = subpasses.data();
    renderPassInfo.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    return m_device->createRenderPassUnique(renderPassInfo);
}

vk::UniquePipelineLayout ou::GraphicsContext::makePipelineLayout(std::vector<vk::DescriptorSetLayout> const& descriptorSetLayouts) const
{
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = static_cast<std::uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    return m_device->createPipelineLayoutUnique(pipelineLayoutInfo);
}

static std::vector<char> readFile(const char* fileName)
{
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file");
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

vk::UniquePipeline ou::GraphicsContext::makePipeline(vk::PipelineLayout pipelineLayout, vk::Extent2D swapExtent,
    vk::RenderPass renderPass, uint32_t subpassIndex, vk::SampleCountFlagBits sampleCount,
    const char* vertexShaderFile, const char* fragmentShaderFile, const char* tcShaderFile, const char* teShaderFile, const char* geometryShaderFile,
    vk::PrimitiveTopology primitiveType,
    vk::CullModeFlags cullMode,
    bool enableDepthTest, BlendMode blendMode, bool attachVertexData,
    vk::VertexInputBindingDescription bindingDescription,
    const std::vector<vk::VertexInputAttributeDescription>& attributeDescriptions,
    const char* vertexShaderEntryPoint, const char* fragmentShaderEntryPoint,
    const char* tcShaderEntryPoint, const char* teShaderEntryPoint, const char* geometryShaderEntryPoint) const
{
    // make shaders
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

    vk::UniqueShaderModule vertShaderModule = createShaderModule(*m_device, readFile(vertexShaderFile));

    vk::PipelineShaderStageCreateInfo vertexShaderStage;
    vertexShaderStage.stage = vk::ShaderStageFlagBits::eVertex;
    vertexShaderStage.module = vertShaderModule.get();
    vertexShaderStage.pName = vertexShaderEntryPoint;
    shaderStages.push_back(vertexShaderStage);

    vk::UniqueShaderModule fragShaderModule;
    if (fragmentShaderFile) {
        fragShaderModule = createShaderModule(*m_device, readFile(fragmentShaderFile));

        vk::PipelineShaderStageCreateInfo fragmentShaderStage;
        fragmentShaderStage.stage = vk::ShaderStageFlagBits::eFragment;
        fragmentShaderStage.module = fragShaderModule.get();
        fragmentShaderStage.pName = fragmentShaderEntryPoint;
        shaderStages.push_back(fragmentShaderStage);
    }

    vk::UniqueShaderModule geometryShaderModule;
    if (geometryShaderFile) {
        geometryShaderModule = createShaderModule(*m_device, readFile(geometryShaderFile));

        vk::PipelineShaderStageCreateInfo geometryShaderStage;
        geometryShaderStage.stage = vk::ShaderStageFlagBits::eGeometry;
        geometryShaderStage.module = geometryShaderModule.get();
        geometryShaderStage.pName = geometryShaderEntryPoint;
        shaderStages.push_back(geometryShaderStage);
    }

    // add tessellation shaders if they are given
    vk::UniqueShaderModule tcShaderModule;
    vk::UniqueShaderModule teShaderModule;
    if (tcShaderFile && teShaderFile) {
        tcShaderModule = createShaderModule(*m_device, readFile(tcShaderFile));
        teShaderModule = createShaderModule(*m_device, readFile(teShaderFile));

        vk::PipelineShaderStageCreateInfo tcShaderStage;
        tcShaderStage.stage = vk::ShaderStageFlagBits::eTessellationControl;
        tcShaderStage.module = tcShaderModule.get();
        tcShaderStage.pName = tcShaderEntryPoint;
        shaderStages.push_back(tcShaderStage);

        vk::PipelineShaderStageCreateInfo teShaderStage;
        teShaderStage.stage = vk::ShaderStageFlagBits::eTessellationEvaluation;
        teShaderStage.module = teShaderModule.get();
        teShaderStage.pName = teShaderEntryPoint;
        shaderStages.push_back(teShaderStage);
    }

    // build the pipeline
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    if (attachVertexData) {
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    }

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = primitiveType;
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

    rasterizer.cullMode = cullMode;
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
    colorBlendAttachment.blendEnable = blendMode != BlendMode::None;
    colorBlendAttachment.srcColorBlendFactor = blendMode == BlendMode::Additive ? vk::BlendFactor::eOne : vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = blendMode == BlendMode::Additive ? vk::BlendFactor::eSrcAlpha : vk::BlendFactor::eOneMinusSrcAlpha;
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
    depthStencil.depthTestEnable = enableDepthTest;
    depthStencil.depthWriteEnable = enableDepthTest;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    vk::PipelineTessellationStateCreateInfo tessInfo{};
    tessInfo.patchControlPoints = 4;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pTessellationState = &tessInfo;
    pipelineInfo.pDynamicState = nullptr;

    pipelineInfo.layout = pipelineLayout;

    // use this pipeline in the render pass
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpassIndex;

    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = -1;

    return m_device->createGraphicsPipelineUnique(nullptr, pipelineInfo);
}

vk::UniquePipeline ou::GraphicsContext::makeComputePipeline(vk::PipelineLayout pipelineLayout, const char* shaderFile) const
{
    vk::UniqueShaderModule shaderModule = createShaderModule(*m_device, readFile(shaderFile));

    vk::PipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eCompute;
    shaderStageCreateInfo.module = *shaderModule;
    shaderStageCreateInfo.pName = "main";

    vk::ComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    return m_device->createComputePipelineUnique(nullptr, pipelineCreateInfo);
}

std::vector<vk::UniqueCommandBuffer> ou::GraphicsContext::allocateCommandBuffers(uint32_t count) const
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = count;

    return m_device->allocateCommandBuffersUnique(allocInfo);
}

vk::UniqueFramebuffer ou::GraphicsContext::makeFramebuffer(std::vector<vk::ImageView> imageViews, vk::RenderPass renderPass,
    vk::Extent2D extent, std::size_t layerCount) const
{
    vk::FramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.renderPass = renderPass;

    framebufferInfo.attachmentCount = static_cast<std::uint32_t>(imageViews.size());
    framebufferInfo.pAttachments = imageViews.data();

    framebufferInfo.width = extent.width;
    framebufferInfo.height = extent.height;
    framebufferInfo.layers = static_cast<std::uint32_t>(layerCount);

    return m_device->createFramebufferUnique(framebufferInfo);
}

std::vector<vk::UniqueSemaphore> ou::GraphicsContext::makeSemaphores(uint32_t count) const
{
    std::vector<vk::UniqueSemaphore> semaphores(count);
    for (auto& semaphore : semaphores) {
        semaphore = m_device->createSemaphoreUnique({});
    }
    return semaphores;
}

std::vector<vk::UniqueFence> ou::GraphicsContext::makeFences(uint32_t count, bool signaled) const
{
    std::vector<vk::UniqueFence> fences(count);
    for (auto& fence : fences) {
        vk::FenceCreateInfo fenceInfo{};
        if (signaled) {
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
        }

        fence = m_device->createFenceUnique(fenceInfo);
    }
    return fences;
}

vk::UniqueBuffer ou::GraphicsContext::makeBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::SharingMode sharingMode) const
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = sharingMode;

    return m_device->createBufferUnique(bufferInfo);
}

vk::UniqueDeviceMemory ou::GraphicsContext::allocateBufferMemory(vk::Buffer buffer, vk::MemoryPropertyFlags properties) const
{
    vk::MemoryRequirements memRequirements = m_device->getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, properties);

    vk::UniqueDeviceMemory bufferMemory = m_device->allocateMemoryUnique(allocInfo);

    m_device->bindBufferMemory(buffer, bufferMemory.get(), 0);

    return bufferMemory;
}

void ou::GraphicsContext::updateMemory(vk::DeviceMemory memory, void* ptr, std::size_t size) const
{
    void* memPtr = m_device->mapMemory(memory, 0, size);
    std::memcpy(memPtr, ptr, size);
    m_device->unmapMemory(memory);
}

void ou::GraphicsContext::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const
{
    auto commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer->copyBuffer(srcBuffer, dstBuffer, { copyRegion });
}

void ou::GraphicsContext::copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, uint32_t width, uint32_t height) const
{
    auto commandBuffer = beginSingleTimeCommands();

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
}

ou::BufferObject ou::GraphicsContext::constructDeviceLocalBuffer(vk::BufferUsageFlags usageFlags, const void* bufferData,
    std::size_t bufferSize) const
{
    // create staging buffer
    vk::UniqueBuffer stagingBuffer = makeBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(*stagingBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // fill staging buffer
    void* const data = m_device->mapMemory(*stagingBufferMemory, 0, bufferSize);
    std::memcpy(data, bufferData, bufferSize);
    m_device->unmapMemory(*stagingBufferMemory);

    // make vertex buffer
    BufferObject result;
    result.buffer = makeBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | usageFlags);
    result.memory = allocateBufferMemory(*result.buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // copy staging -> vertex buffer
    copyBuffer(*stagingBuffer, *result.buffer, bufferSize);

    return result;
}

ou::BufferObject ou::GraphicsContext::makeHostVisibleBuffer(vk::BufferUsageFlags usageFlags, std::size_t bufferSize) const
{
    BufferObject buf;
    buf.buffer = makeBuffer(bufferSize, usageFlags);
    buf.memory = allocateBufferMemory(*buf.buffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    return buf;
}

void ou::GraphicsContext::generateMipmaps(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format, vk::Extent2D extent, uint32_t mipLevels) const
{
    // check if image format can generate linear filter mipmaps
    vk::FormatProperties formatProperties = m_physicalDevice.getFormatProperties(format);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting");
    }

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

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
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

        commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
            { blit }, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {}, { barrier });
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });
}

ou::ImageObject ou::GraphicsContext::makeTextureImage(const char* filename) const
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

    vk::UniqueBuffer stagingBuffer = makeBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::UniqueDeviceMemory stagingBufferMemory = allocateBufferMemory(*stagingBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // copy image data to buffer
    void* const data = m_device->mapMemory(*stagingBufferMemory, 0, imageSize);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    m_device->unmapMemory(*stagingBufferMemory);

    stbi_image_free(pixels);

    // create image object
    const vk::Format imageFormat = vk::Format::eR8G8B8A8Unorm;
    ImageObject texture = makeImage(vk::SampleCountFlagBits::e1, mipLevels, imageExtent, 1, imageFormat,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::ImageAspectFlagBits::eColor);

    // image layout is not gpu accessible by now
    transitionImageLayout(*beginSingleTimeCommands(),
        *texture.image, 1, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);

    // copy buffer to image
    copyBufferToImage(*stagingBuffer, *texture.image, imageExtent.width, imageExtent.height);

    // generate mipmaps
    generateMipmaps(*beginSingleTimeCommands(), *texture.image, imageFormat, imageExtent, mipLevels);

    return texture;
}

vk::UniqueSampler ou::GraphicsContext::makeTextureSampler(bool unnormalizedCoordinates) const
{
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;

    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp = vk::CompareOp::eLess;

    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;

    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;

    if (unnormalizedCoordinates) {
        samplerInfo.unnormalizedCoordinates = true;

        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 16;
    } else {
        samplerInfo.unnormalizedCoordinates = false;

        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 12.0f;

        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;
    }

    return m_device->createSamplerUnique(samplerInfo);
}

vk::SampleCountFlagBits ou::GraphicsContext::getMaxUsableSampleCount(uint32_t preferredSampleCount) const
{
    vk::PhysicalDeviceProperties physicalDeviceProperties = m_physicalDevice.getProperties();

    vk::SampleCountFlags counts = static_cast<vk::SampleCountFlags>(std::min(
        static_cast<std::uint32_t>(physicalDeviceProperties.limits.framebufferColorSampleCounts),
        static_cast<std::uint32_t>(physicalDeviceProperties.limits.framebufferDepthSampleCounts)));

    if (counts & vk::SampleCountFlagBits::e64 && preferredSampleCount >= 64) {
        return vk::SampleCountFlagBits::e64;
    }
    if (counts & vk::SampleCountFlagBits::e32 && preferredSampleCount >= 32) {
        return vk::SampleCountFlagBits::e32;
    }
    if (counts & vk::SampleCountFlagBits::e16 && preferredSampleCount >= 16) {
        return vk::SampleCountFlagBits::e16;
    }
    if (counts & vk::SampleCountFlagBits::e8 && preferredSampleCount >= 8) {
        return vk::SampleCountFlagBits::e8;
    }
    if (counts & vk::SampleCountFlagBits::e4 && preferredSampleCount >= 4) {
        return vk::SampleCountFlagBits::e4;
    }
    if (counts & vk::SampleCountFlagBits::e2 && preferredSampleCount >= 2) {
        return vk::SampleCountFlagBits::e2;
    }
    return vk::SampleCountFlagBits::e1;
}

ou::SingleTimeCommandBuffer::SingleTimeCommandBuffer(vk::UniqueCommandBuffer&& commandBuf, const vk::Queue& queue)
    : commandBuffer(std::move(commandBuf))
    , queue(queue)
{
}

ou::SingleTimeCommandBuffer::SingleTimeCommandBuffer(ou::SingleTimeCommandBuffer&& other)
    : commandBuffer(std::move(other.commandBuffer))
    , queue(other.queue)
{
}

ou::SingleTimeCommandBuffer& ou::SingleTimeCommandBuffer::operator=(ou::SingleTimeCommandBuffer&& other)
{
    commandBuffer = std::move(other.commandBuffer);
    queue = other.queue;
    return *this;
}

vk::CommandBuffer ou::SingleTimeCommandBuffer::operator*() const
{
    return *commandBuffer;
}

const vk::CommandBuffer* ou::SingleTimeCommandBuffer::operator->() const
{
    return &*commandBuffer;
}

ou::SingleTimeCommandBuffer::~SingleTimeCommandBuffer()
{
    commandBuffer->end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandBuffer;

    queue.submit({ submitInfo }, nullptr);
    queue.waitIdle();
}
