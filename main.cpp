#include "main.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>

namespace ou {

static const std::size_t maxFramesInFlight = 2;

struct Vertex {
    glm::vec3 pos;

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
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions(1);
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        return attributeDescriptions;
    }
};

ModelObject::ModelObject(GraphicsContext const& context,
    const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices)
    : vertexBuffer(context.constructDeviceLocalBuffer(vk::BufferUsageFlagBits::eVertexBuffer,
          vertices.data(), vertices.size() * sizeof(vertices[0])))
    , indexBuffer(context.constructDeviceLocalBuffer(vk::BufferUsageFlagBits::eIndexBuffer,
          indices.data(), indices.size() * sizeof(indices[0])))
    , indexCount(indices.size())
{
}

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 eyePos;
};

SwapchainObject::SwapchainObject(const GraphicsContext& context, vk::DescriptorSetLayout descriptorSetLayout,
    SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain)
{
    // make swapchain
    swapchain = context.makeSwapchain(properties, oldSwapchain);
    swapchainImages = context.retrieveSwapchainImages(*swapchain);
    const auto swapchainImageCount = static_cast<std::uint32_t>(swapchainImages.size());
    for (const auto& image : swapchainImages) {
        swapchainImageViews.push_back(context.makeImageView(image,
            properties.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
    }

    // make multisampling buffer
    const vk::SampleCountFlagBits sampleCount = context.getMaxUsableSampleCount(2);
    multiSampleImage = context.makeMultiSampleImage(properties.surfaceFormat.format, properties.extent, sampleCount);

    // make depth buffer
    depthImage = context.makeDepthImage(properties.extent, sampleCount);

    // make render pass
    renderPass = context.makeRenderPass(sampleCount, properties.surfaceFormat.format, depthImage.format, 3);

    // make pipelines
    pipelineLayout = context.makePipelineLayout(descriptorSetLayout);
    terrainPipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 0, sampleCount,
        "shaders/planet.vert.spv", "shaders/planet.frag.spv", "shaders/planet.tesc.spv", "shaders/planet.tese.spv",
        vk::PrimitiveTopology::ePatchList,
        Vertex::getBindingDescription(), Vertex::getAttributeDescriptions());

    oceanPipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 1, sampleCount,
        "shaders/water.vert.spv", "shaders/water.frag.spv", "shaders/water.tesc.spv", "shaders/water.tese.spv",
        vk::PrimitiveTopology::ePatchList,
        Vertex::getBindingDescription(), Vertex::getAttributeDescriptions());

    atmospherePipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 2, sampleCount,
        "shaders/air.vert.spv", "shaders/air.frag.spv", "shaders/air.tesc.spv", "shaders/air.tese.spv",
        vk::PrimitiveTopology::ePatchList,
        Vertex::getBindingDescription(), Vertex::getAttributeDescriptions());

    // make command buffers
    commandBuffers = context.allocateCommandBuffers(swapchainImageCount);

    // make framebuffers
    framebuffers = context.makeFramebuffers(swapchainImageViews, *depthImage.view,
        *multiSampleImage.view, *renderPass, properties.extent);
}

VulkanApplication::VulkanApplication()
    : m_context(600, 480, false)

    // figure out swapchain properties
    , m_swapchainProps(m_context.selectSwapchainProperties())

    // make description set layout, pool, and sets
    , m_descriptorSetLayout(m_context.makeDescriptorSetLayout())
    , m_descriptorPool(m_context.makeDescriptorPool(m_swapchainProps.imageCount))
    , m_descriptorSets(m_context.makeDescriptorSets(*m_descriptorPool, *m_descriptorSetLayout, m_swapchainProps.imageCount))

    // make swapchain
    , m_swapchain(m_context, *m_descriptorSetLayout, m_swapchainProps)

    // make semaphores for synchronizing frame drawing operations
    , m_imageAvailableSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_renderFinishedSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_inFlightFences(m_context.makeFences(maxFramesInFlight))

    // make textures
    , m_textureImage(m_context.makeTextureImage("textures/texture.jpg"))

    // make sampler
    , m_sampler(m_context.makeTextureSampler())
{
    std::vector<glm::vec3> vertices;
    std::vector<std::uint16_t> indices;

    // convert spherical coords to cartesian
    auto spherical = [](float theta, float phi) {
        return glm::vec3({ std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta) });
    };

    const int meridians = 90;
    const int parallels = 45;
    const float theta = glm::pi<float>() / parallels;
    const float phi = 2 * glm::pi<float>() / meridians;

    for (int p = 0; p <= parallels; ++p) {
        for (int m = 0; m < meridians; ++m) {
            vertices.push_back(spherical(theta * p, phi * m));
            if (m > 0) {
                indices.push_back(std::uint16_t(std::max(0, int(vertices.size()) - meridians - 1)));
                indices.push_back(std::uint16_t(std::max(0, int(vertices.size()) - meridians - 2)));
                indices.push_back(std::uint16_t(int(vertices.size()) - 2));
                indices.push_back(std::uint16_t(int(vertices.size()) - 1));
            }
        }
        indices.push_back(std::uint16_t(std::max(0, int(vertices.size()) - meridians - meridians)));
        indices.push_back(std::uint16_t(std::max(0, int(vertices.size()) - meridians - 1)));
        indices.push_back(std::uint16_t(int(vertices.size() - 1)));
        indices.push_back(std::uint16_t(int(vertices.size() - meridians)));
    }

    std::vector<Vertex> attributes;
    for (auto const& vertex : vertices) {
        attributes.push_back({ vertex });
    }
    std::cout << "vertex count: " << indices.size() << std::endl;

    m_model = ModelObject(m_context, attributes, indices);

    // make uniform buffers
    for (std::uint32_t i = 0; i < m_swapchainProps.imageCount; ++i) {
        m_uniformBuffers.push_back(m_context.makeBuffer(sizeof(UniformBufferObject), vk::BufferUsageFlagBits::eUniformBuffer));
        m_uniformBuffersMemory.push_back(m_context.allocateBufferMemory(*m_uniformBuffers.back(),
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    }

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();

    // recreate swapchain
    m_swapchainProps = m_context.selectSwapchainProperties();
    m_swapchain = SwapchainObject(m_context, *m_descriptorSetLayout, m_swapchainProps, *m_swapchain.swapchain);

    // re-record draw commands
    recordDrawCommands();
}

void VulkanApplication::recordDrawCommands()
{
    std::size_t index = 0;
    for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.commandBuffers) {
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        beginInfo.pInheritanceInfo = nullptr;

        commandBuffer->begin(beginInfo);

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = *m_swapchain.renderPass;
        renderPassInfo.framebuffer = *m_swapchain.framebuffers[index];
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
        {
            commandBuffer->bindVertexBuffers(0, { *m_model.vertexBuffer.buffer }, { 0 });

            commandBuffer->bindIndexBuffer(*m_model.indexBuffer.buffer, 0, vk::IndexType::eUint16);

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

            commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.pipelineLayout, 0,
                { m_descriptorSets[index] }, {});

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.terrainPipeline);
            commandBuffer->drawIndexed(static_cast<std::uint32_t>(m_model.indexCount), 1, 0, 0, 0);

            commandBuffer->nextSubpass(vk::SubpassContents::eInline);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.oceanPipeline);
            commandBuffer->drawIndexed(static_cast<std::uint32_t>(m_model.indexCount), 1, 0, 0, 0);

            commandBuffer->nextSubpass(vk::SubpassContents::eInline);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.atmospherePipeline);
            commandBuffer->drawIndexed(static_cast<std::uint32_t>(m_model.indexCount), 1, 0, 0, 0);
        }
        commandBuffer->endRenderPass();

        commandBuffer->end();

        index++;
    }
}

void VulkanApplication::run()
{
    using namespace std::chrono;
    m_startTime = m_lastFpsTime = m_lastFrameTime = system_clock::now();
    m_fpsFrameCounter = m_fpsMeasurementsCount = 0;

    // main loop
    while (!glfwWindowShouldClose(m_context.window())) {
        drawFrame();

        glfwPollEvents();

        // calculate FPS
        m_fpsFrameCounter++;
        const auto currentTime = system_clock::now();
        const auto elapsedTime = currentTime - m_lastFpsTime;

        if (elapsedTime >= seconds(1)) {
            double fps = static_cast<double>(m_fpsFrameCounter) / duration_cast<seconds>(elapsedTime).count();
            m_averageFps = (m_averageFps * m_fpsMeasurementsCount + fps) / (m_fpsMeasurementsCount + 1);

            m_lastFpsTime = currentTime;
            m_fpsFrameCounter = 0;
            m_fpsMeasurementsCount++;
        }

        const double frameInterval = 1.0 / (m_context.refreshRate() + 1);
        while (system_clock::now() - m_lastFrameTime < duration<double>(frameInterval))
            ;
        m_lastFrameTime = currentTime;
    }

    std::cout << "Average fps: " << std::setprecision(0) << std::fixed << m_averageFps << std::endl;
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
    m_context.device().acquireNextImageKHR(*m_swapchain.swapchain, timeOut,
        *m_imageAvailableSemaphores[m_currentFrame], nullptr, &imageIndex);

    // update the uniform data
    {
        using namespace std::chrono;
        float time = duration<float, seconds::period>(system_clock::now() - m_startTime).count();
        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time / 8.0f * glm::radians(90.0f), glm::normalize(glm::vec3(0.0f, 0.3f, 0.8f)));
        ubo.model = glm::scale(ubo.model, glm::vec3(1.0f, 1.0f, 1.0f));

        ubo.eyePos = glm::rotate(glm::mat4(1.0f), time / 4.0f * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f))
            * glm::vec4(2.0f, 2.0f, 2.0f, 1.0f);

        ubo.view = glm::lookAt(ubo.eyePos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

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

    std::array<vk::CommandBuffer, 1> commandBuffers = { { *m_swapchain.commandBuffers[imageIndex] } };
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

    std::array<vk::SwapchainKHR, 1> swapchains = { { *m_swapchain.swapchain } };
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
