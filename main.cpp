#include "main.h"

#include <iomanip>
#include <iostream>

namespace ou {

static const std::size_t maxFramesInFlight = 2;

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

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions()
    {
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions(3);
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

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
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
        m_swapchainImageViews.push_back(makeImageView(m_device.get(), image, m_swapchainProps.surfaceFormat.format,
            vk::ImageAspectFlagBits::eColor, 1));
    }

    // make command pool and command buffers
    m_commandPool = makeCommandPool(m_device.get(), queueFamilies.graphicsFamilyIndex);
    m_commandBuffers = allocateCommandBuffers(m_device.get(), m_commandPool.get(), swapchainImageCount);

    // make depth buffer
    ImageObject depth = makeDepthImage(physicalDevice, m_device.get(), m_commandPool.get(),
        m_graphicsQueue, m_swapchainProps.extent);
    m_depthImage = std::move(depth.image);
    m_depthImageMemory = std::move(depth.imageMemory);
    m_depthImageView = std::move(depth.imageView);

    // make render pass
    m_renderPass = makeRenderPass(m_device.get(), m_swapchainProps.surfaceFormat.format, depth.format);

    // make description set layout, pool, and sets
    m_descriptorSetLayout = makeDescriptorSetLayout(m_device.get());
    m_descriptorPool = makeDescriptorPool(m_device.get(), swapchainImageCount);
    m_descriptorSets = makeDescriptorSets(m_device.get(), m_descriptorPool.get(),
        m_descriptorSetLayout.get(), swapchainImageCount);

    // make pipeline
    m_pipelineLayout = makePipelineLayout(m_device.get(), m_descriptorSetLayout.get());
    m_graphicsPipeline = makePipeline(m_device.get(), m_pipelineLayout.get(), m_swapchainProps.extent, m_renderPass.get(),
        Vertex::getBindingDescription(), Vertex::getAttributeDescriptions());

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
        m_graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertices.data(), vertices.size() * sizeof(vertices[0]));
    m_vertexBuffer = std::move(vertexBuffer.buffer);
    m_vertexBufferMemory = std::move(vertexBuffer.bufferMemory);

    // make & fill index buffer
    BufferObject indexBuffer = constructDeviceLocalBuffer(physicalDevice, m_device.get(), m_commandPool.get(),
        m_graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indices.data(), indices.size() * sizeof(indices[0]));
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
    m_textureImageView = std::move(texture.imageView);

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
        ubo.model = glm::scale(ubo.model, glm::vec3(0.2f, 0.2f, 0.2f));
        ubo.view = glm::lookAt(glm::vec3(0.3f, 0.3f, ((glm::sin(time) + 1.0f) / 2.0f) * 5.0f),
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

} // namespace ou

int main()
{
    try {
        ou::VulkanApplication app;
        app.run();
    } catch (std::exception const& err) {
        std::cerr << "Error while running the program: " << err.what() << std::endl;
    }

    return 0;
}
