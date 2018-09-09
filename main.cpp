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

SwapchainObject::SwapchainObject(const GraphicsContext& context,
    vk::DescriptorSetLayout descriptorSetLayout, SwapchainProperties const& properties)
{
    // make swapchain
    m_swapchain = context.makeSwapchain(properties);
    m_swapchainImages = context.retrieveSwapchainImages(*m_swapchain);
    const auto swapchainImageCount = static_cast<std::uint32_t>(m_swapchainImages.size());
    for (const auto& image : m_swapchainImages) {
        m_swapchainImageViews.push_back(context.makeImageView(image,
            properties.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
    }

    // make multisampling buffer
    const vk::SampleCountFlagBits sampleCount = context.getMaxUsableSampleCount(4);
    m_multiSampleImage = context.makeMultiSampleImage(properties.surfaceFormat.format, properties.extent, sampleCount);

    // make depth buffer
    m_depthImage = context.makeDepthImage(properties.extent, sampleCount);

    // make render pass
    m_renderPass = context.makeRenderPass(sampleCount, properties.surfaceFormat.format, m_depthImage.format);

    // make pipeline
    m_pipelineLayout = context.makePipelineLayout(descriptorSetLayout);
    m_graphicsPipeline = context.makePipeline(*m_pipelineLayout, properties.extent, *m_renderPass,
        sampleCount, Vertex::getBindingDescription(), Vertex::getAttributeDescriptions());

    // make command buffers
    m_commandBuffers = context.allocateCommandBuffers(swapchainImageCount);

    // make framebuffers
    m_framebuffers = context.makeFramebuffers(m_swapchainImageViews, *m_depthImage.view,
        *m_multiSampleImage.view, *m_renderPass, properties.extent);
}

VulkanApplication::VulkanApplication()
{
    // figure out swapchain image count
    m_swapchainProps = m_context.selectSwapchainProperties();
    const std::uint32_t swapchainImageCount = m_swapchainProps.minImageCount;

    // make description set layout, pool, and sets
    m_descriptorSetLayout = m_context.makeDescriptorSetLayout();
    m_descriptorPool = m_context.makeDescriptorPool(swapchainImageCount);
    m_descriptorSets = m_context.makeDescriptorSets(*m_descriptorPool, *m_descriptorSetLayout, swapchainImageCount);

    // make swapchain
    m_swapchain = SwapchainObject(m_context, *m_descriptorSetLayout, m_swapchainProps);

    // make semaphores for synchronizing frame drawing operations
    for (std::size_t index = 0; index < maxFramesInFlight; ++index) {
        m_imageAvailableSemaphores.push_back(m_context.makeSemaphore());
        m_renderFinishedSemaphores.push_back(m_context.makeSemaphore());
        m_inFlightFences.push_back(m_context.makeFence());
    }

    // make & fill vertex buffer
    m_vertexBuffer = m_context.constructDeviceLocalBuffer(vk::BufferUsageFlagBits::eVertexBuffer,
        vertices.data(), vertices.size() * sizeof(vertices[0]));

    // make & fill index buffer
    m_indexBuffer = m_context.constructDeviceLocalBuffer(vk::BufferUsageFlagBits::eIndexBuffer,
        indices.data(), indices.size() * sizeof(indices[0]));

    // make uniform buffers
    for (std::uint32_t i = 0; i < swapchainImageCount; ++i) {
        m_uniformBuffers.push_back(m_context.makeBuffer(sizeof(UniformBufferObject), vk::BufferUsageFlagBits::eUniformBuffer));
        m_uniformBuffersMemory.push_back(m_context.allocateBufferMemory(*m_uniformBuffers.back(),
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    }

    // make textures
    m_textureImage = m_context.makeTextureImage("textures/texture.jpg");

    // make sampler
    m_sampler = m_context.makeTextureSampler();

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();

    // recreate swapchain
    m_swapchainProps = m_context.selectSwapchainProperties();
    m_swapchain = SwapchainObject(m_context, *m_descriptorSetLayout, m_swapchainProps);

    // re-record draw commands
    recordDrawCommands();
}

void VulkanApplication::recordDrawCommands()
{
    std::size_t index = 0;
    for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.m_commandBuffers) {
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        beginInfo.pInheritanceInfo = nullptr;

        commandBuffer->begin(beginInfo);

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = *m_swapchain.m_renderPass;
        renderPassInfo.framebuffer = *m_swapchain.m_framebuffers[index];
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

        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.m_graphicsPipeline);

        commandBuffer->bindVertexBuffers(0, { *m_vertexBuffer.buffer }, { 0 });

        commandBuffer->bindIndexBuffer(*m_indexBuffer.buffer, 0, vk::IndexType::eUint16);

        // bind uniform descriptor sets
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *m_uniformBuffers[index];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = *m_textureImage.view;
        imageInfo.sampler = *m_sampler;

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].dstSet = m_descriptorSets[index];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = m_descriptorSets[index];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        m_context.device().updateDescriptorSets(descriptorWrites, {});

        commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.m_pipelineLayout, 0,
            { m_descriptorSets[index] }, {});

        commandBuffer->drawIndexed(static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer->endRenderPass();

        commandBuffer->end();

        index++;
    }
}

void VulkanApplication::run()
{
    using namespace std::chrono;
    m_startTime = m_lastFpsTimePoint = system_clock::now();
    m_fpsCounter = 0;

    // main loop
    while (!glfwWindowShouldClose(m_context.window())) {
        glfwPollEvents();

        drawFrame();

        // calculate FPS
        m_fpsCounter++;
        const auto currentTime = system_clock::now();
        const auto elapsedTime = currentTime - m_lastFpsTimePoint;

        if (elapsedTime >= seconds(1)) {
            double fps = static_cast<double>(m_fpsCounter) / duration_cast<seconds>(elapsedTime).count();
            std::cout << "FPS: " << std::fixed << std::setprecision(0) << fps << "\n"; // std::endl;

            m_lastFpsTimePoint = currentTime;
            m_fpsCounter = 0;
        }
    }

    m_context.device().waitIdle();
}

void VulkanApplication::drawFrame()
{
    // The drawFrame function will perform the following operations:
    // 1. Acquire an image from the swap chain
    // 2. Execute the command buffer with that image as attachment in the framebuffer
    // 3. Return the image to the swap chain for presentation

    const std::uint64_t timeOut = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

    std::array<vk::Fence, 1> inFlightFences = { { *m_inFlightFences[m_currentFrame] } };
    m_context.device().waitForFences(inFlightFences, VK_TRUE, timeOut);

    // acquire next image to write into from the swap chain
    // note: this function is asynchronous
    std::uint32_t imageIndex;
    m_context.device().acquireNextImageKHR(*m_swapchain.m_swapchain, timeOut,
        *m_imageAvailableSemaphores[m_currentFrame], nullptr, &imageIndex);

    // update the uniform data
    {
        using namespace std::chrono;
        float time = duration<float, seconds::period>(system_clock::now() - m_startTime).count();
        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.model = glm::scale(ubo.model, glm::vec3(0.2f, 0.2f, 0.2f));
        ubo.view = glm::lookAt(glm::vec3(0.3f, 0.3f, 0.3f),
            glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(m_swapchainProps.extent.width) / m_swapchainProps.extent.height,
            0.1f, 10.0f);
        ubo.proj[1][1] *= -1; // invert Y axis

        void* const data = m_context.device().mapMemory(*m_uniformBuffersMemory[imageIndex], 0, sizeof(ubo));
        std::memcpy(data, &ubo, sizeof(ubo));
        m_context.device().unmapMemory(*m_uniformBuffersMemory[imageIndex]);
    }

    // execute the command buffer
    vk::SubmitInfo submitInfo{};

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    std::array<vk::Semaphore, 1> imageAvailableSemaphores = { { *m_imageAvailableSemaphores[m_currentFrame] } };
    submitInfo.waitSemaphoreCount = imageAvailableSemaphores.size();
    submitInfo.pWaitSemaphores = imageAvailableSemaphores.data();
    submitInfo.pWaitDstStageMask = &waitStage;

    std::array<vk::CommandBuffer, 1> commandBuffers = { { *m_swapchain.m_commandBuffers[imageIndex] } };
    submitInfo.commandBufferCount = commandBuffers.size();
    submitInfo.pCommandBuffers = commandBuffers.data();

    std::array<vk::Semaphore, 1> renderFinishedSemaphores = { { *m_renderFinishedSemaphores[m_currentFrame] } };
    submitInfo.signalSemaphoreCount = renderFinishedSemaphores.size();
    submitInfo.pSignalSemaphores = renderFinishedSemaphores.data();

    m_context.device().resetFences(inFlightFences);
    m_context.graphicsQueue().submit({ submitInfo }, *m_inFlightFences[m_currentFrame]);

    // write back to the swap chain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = renderFinishedSemaphores.size();
    presentInfo.pWaitSemaphores = renderFinishedSemaphores.data();

    std::array<vk::SwapchainKHR, 1> swapchains = { { *m_swapchain.m_swapchain } };
    presentInfo.swapchainCount = swapchains.size();
    presentInfo.pSwapchains = swapchains.data();
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr;

    vk::Result result = m_context.presentQueue().presentKHR(&presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        refreshSwapchain();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swapchain image");
    }

    m_currentFrame = (m_currentFrame + 1) % maxFramesInFlight;
}

} // namespace ou

int main()
{
    try {
        ou::VulkanApplication app;
        app.run();
    } catch (std::exception const& err) {
        std::cerr << "Exception raised while running the program: " << err.what() << std::endl;
    }

    return 0;
}
