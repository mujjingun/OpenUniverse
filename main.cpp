#include "main.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/intersect.hpp"
#include "glm/gtx/projection.hpp"
#include "glm/gtx/string_cast.hpp"

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
    glm::vec4 eyePos;
    glm::vec4 modelEyePos;
    glm::vec4 lightDir;
    std::int32_t parallelCount;
    std::int32_t meridianCount;
};

SwapchainObject::SwapchainObject(const GraphicsContext& context, SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain)
{
    // make swapchain
    swapchain = context.makeSwapchain(properties, oldSwapchain);
    swapchainImages = context.retrieveSwapchainImages(*swapchain);
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
    renderPass = context.makeRenderPass(sampleCount, properties.surfaceFormat.format, depthImage.format, 2);

    // make descriptor sets
    descriptorSet = context.makeDescriptorSet(properties.imageCount,
        { vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eCombinedImageSampler, vk::DescriptorType::eUniformBuffer },
        { vk::ShaderStageFlagBits::eAll,
            vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
            vk::ShaderStageFlagBits::eAll });

    // make pipelines
    pipelineLayout = context.makePipelineLayout(*descriptorSet.layout);
    terrainPipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 0, sampleCount,
        "shaders/planet.vert.spv", "shaders/planet.frag.spv", "shaders/planet.tesc.spv", "shaders/planet.tese.spv",
        vk::PrimitiveTopology::ePatchList, false, {}, {});

    atmospherePipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 1, sampleCount,
        "shaders/air.vert.spv", "shaders/air.frag.spv", "shaders/air.tesc.spv", "shaders/air.tese.spv",
        vk::PrimitiveTopology::ePatchList, false, {}, {});

    // make command buffers
    commandBuffers = context.allocateCommandBuffers(properties.imageCount);

    // make framebuffers
    for (auto const& swapchainImageView : swapchainImageViews) {
        framebuffers.push_back(context.makeFramebuffer(*swapchainImageView, *depthImage.view,
            *multiSampleImage.view, *renderPass, properties.extent));
    }

    // make offscreen render target
    const std::size_t noiseFrameBuffersCount = 1;
    const vk::SampleCountFlagBits noiseSampleCount = context.getMaxUsableSampleCount(4);
    const vk::Extent2D noiseImageExtent = { properties.extent.width * 2, properties.extent.height * 2 };
    noiseImage = context.makeImage(vk::SampleCountFlagBits::e1, 1, noiseImageExtent, vk::Format::eR32G32B32A32Sfloat,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::ImageAspectFlagBits::eColor);
    noiseDepthImage = context.makeDepthImage(noiseImageExtent, noiseSampleCount);
    noiseMultiSampleImage = context.makeMultiSampleImage(noiseImage.format, noiseImageExtent, noiseSampleCount);

    noiseRenderPass = context.makeRenderPass(noiseSampleCount, noiseImage.format, depthImage.format, 1);

    noiseDescriptorSet = context.makeDescriptorSet(noiseFrameBuffersCount,
        { vk::DescriptorType::eUniformBuffer },
        { vk::ShaderStageFlagBits::eAll });

    noisePipelineLayout = context.makePipelineLayout(*noiseDescriptorSet.layout);
    noisePipeline = context.makePipeline(*noisePipelineLayout, noiseImageExtent, *noiseRenderPass, 0, noiseSampleCount,
        "shaders/noise.vert.spv", "shaders/noise.frag.spv", nullptr, nullptr,
        vk::PrimitiveTopology::eTriangleFan, false, {}, {});

    noiseCommandBuffers = context.allocateCommandBuffers(noiseFrameBuffersCount);

    for (std::size_t i = 0; i < noiseFrameBuffersCount; ++i) {
        noiseFramebuffers.push_back(context.makeFramebuffer(*noiseImage.view, *noiseDepthImage.view,
            *noiseMultiSampleImage.view, *noiseRenderPass, noiseImageExtent));
    }
}

VulkanApplication::VulkanApplication()
    : m_context(600, 480, false)

    // figure out swapchain properties
    , m_swapchainProps(m_context.selectSwapchainProperties())

    // make swapchain
    , m_swapchain(m_context, m_swapchainProps)

    // make semaphores for synchronizing frame drawing operations
    , m_imageAvailableSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_renderFinishedSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_offscreenPassSemaphore(std::move(m_context.makeSemaphores(1)[0]))
    , m_inFlightFences(m_context.makeFences(maxFramesInFlight))

    // make textures
    , m_textureImage(m_context.makeTextureImage("textures/texture.jpg"))

    // make sampler
    , m_sampler(m_context.makeTextureSampler(false))

    , m_parallelCount(200)
    , m_meridianCount(200)
    , m_renderHeightmap(true)
{
    // make uniform buffers
    for (std::uint32_t i = 0; i < m_swapchainProps.imageCount; ++i) {
        m_uniformBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UniformBufferObject)));
    }
    m_mapBoundsUniformBuffer = m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MapBoundsObject));

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();
    m_renderHeightmap = true;

    // recreate swapchain
    m_swapchainProps = m_context.selectSwapchainProperties();
    m_swapchain = SwapchainObject(m_context, m_swapchainProps, *m_swapchain.swapchain);

    // re-record draw commands
    recordDrawCommands();
}

void VulkanApplication::recordDrawCommands()
{
    std::array<vk::ClearValue, 2> clearValues{};

    // color buffer clear value
    clearValues[0].color.float32[0] = 0.0f;
    clearValues[0].color.float32[1] = 0.0f;
    clearValues[0].color.float32[2] = 0.0f;
    clearValues[0].color.float32[3] = 1.0f;

    // depth buffer clear value
    clearValues[1].depthStencil.depth = 1.0f;
    clearValues[1].depthStencil.stencil = 0;

    // record offscreen rendering
    {
        std::size_t index = 0;
        for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.noiseCommandBuffers) {
            vk::CommandBufferBeginInfo beginInfo{};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
            beginInfo.pInheritanceInfo = nullptr;

            commandBuffer->begin(beginInfo);

            // bind uniform descriptor sets
            std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};

            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = *m_mapBoundsUniformBuffer.buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites[0].dstSet = m_swapchain.noiseDescriptorSet.sets[index];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            m_context.device().updateDescriptorSets(descriptorWrites, {});

            vk::RenderPassBeginInfo noiseRenderPassInfo{};
            noiseRenderPassInfo.renderPass = *m_swapchain.noiseRenderPass;
            noiseRenderPassInfo.framebuffer = *m_swapchain.noiseFramebuffers[index];
            noiseRenderPassInfo.renderArea.offset.x = 0;
            noiseRenderPassInfo.renderArea.offset.y = 0;
            noiseRenderPassInfo.renderArea.extent = m_swapchain.noiseImage.extent;
            noiseRenderPassInfo.clearValueCount = clearValues.size();
            noiseRenderPassInfo.pClearValues = clearValues.data();

            commandBuffer->beginRenderPass(noiseRenderPassInfo, vk::SubpassContents::eInline);
            {
                commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.noisePipelineLayout, 0,
                    { m_swapchain.noiseDescriptorSet.sets[index] }, {});

                commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.noisePipeline);
                commandBuffer->draw(4, 1, 0, 0);
            }
            commandBuffer->endRenderPass();

            commandBuffer->end();

            index++;
        }
    }

    // record main rendering
    {
        std::size_t index = 0;
        for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.commandBuffers) {

            // bind uniform descriptor sets
            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};

            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = *m_uniformBuffers[index].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites[0].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageInfo.imageView = *m_swapchain.noiseImage.view;
            imageInfo.sampler = *m_sampler;

            descriptorWrites[1].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vk::DescriptorBufferInfo mapBoundsUniformBufferInfo{};
            mapBoundsUniformBufferInfo.buffer = *m_mapBoundsUniformBuffer.buffer;
            mapBoundsUniformBufferInfo.offset = 0;
            mapBoundsUniformBufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites[2].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &mapBoundsUniformBufferInfo;

            m_context.device().updateDescriptorSets(descriptorWrites, {});

            // begin recording
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
            renderPassInfo.clearValueCount = clearValues.size();
            renderPassInfo.pClearValues = clearValues.data();

            commandBuffer->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            {
                commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.pipelineLayout, 0,
                    { m_swapchain.descriptorSet.sets[index] }, {});

                commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.terrainPipeline);
                auto vertexCount = static_cast<std::uint32_t>(m_parallelCount * m_meridianCount * 4);
                commandBuffer->draw(vertexCount, 1, 0, 0);

                commandBuffer->nextSubpass(vk::SubpassContents::eInline);

                commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.atmospherePipeline);
                commandBuffer->draw(vertexCount, 1, 0, 0);
            }
            commandBuffer->endRenderPass();

            commandBuffer->end();

            index++;
        }
    }
}

void VulkanApplication::drawFrame()
{
    // The drawFrame function will perform the following operations:
    // 1. Acquire an image from the swap chain
    // 2. Execute the command buffer with that image as attachment in the framebuffer
    // 3. Return the image to the swap chain for presentation

    const std::uint64_t timeOut = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

    vk::Fence inFlightFence = *m_inFlightFences[m_currentFrame];
    m_context.device().waitForFences({ inFlightFence }, VK_TRUE, timeOut);

    // acquire next image to write into from the swap chain
    // note: this function is asynchronous
    std::uint32_t imageIndex = m_context.device()
                                   .acquireNextImageKHR(*m_swapchain.swapchain, timeOut, *m_imageAvailableSemaphores[m_currentFrame], nullptr)
                                   .value;

    // update the uniform data
    {
        // render pass
        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), m_planetRotateAngle, glm::normalize(glm::vec3(0.4f, 0.0f, 0.8f)));
        ubo.model = glm::scale(ubo.model, glm::vec3(1.0f, 1.0f, 1.0f));

        ubo.eyePos = glm::vec4(m_eyePosition, 0);
        ubo.view = glm::lookAt(m_eyePosition, m_eyePosition + m_lookDirection, m_upDirection);

        ubo.modelEyePos = glm::inverse(ubo.model) * ubo.eyePos;
        float near = 0.5f * (length(glm::vec3(ubo.modelEyePos)) - 1.0f);

        ubo.proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(m_swapchainProps.extent.width) / m_swapchainProps.extent.height, near, near * 100.0f);
        ubo.proj[1][1] *= -1; // invert Y axis

        ubo.parallelCount = m_parallelCount;
        ubo.meridianCount = m_meridianCount;

        ubo.lightDir = glm::vec4(glm::normalize(glm::vec3(1, -1, 0)), 0);

        void* memPtr = m_context.device().mapMemory(*m_uniformBuffers[imageIndex].memory, 0, sizeof(ubo));
        std::memcpy(memPtr, &ubo, sizeof(ubo));
        m_context.device().unmapMemory(*m_uniformBuffers[imageIndex].memory);

        // determine map bounds
        float newMapSpan = std::acos(1 / length(ubo.modelEyePos));
        if (newMapSpan < m_currentMapBounds.mapSpanTheta / 2 || newMapSpan > m_currentMapBounds.mapSpanTheta) {
            glm::vec3 normEye = glm::normalize(ubo.modelEyePos);
            m_currentMapBounds.mapCenterTheta = std::acos(normEye.z);
            m_currentMapBounds.mapCenterPhi = std::atan2(normEye.y, normEye.x);
            if (newMapSpan < m_currentMapBounds.mapSpanTheta / 2)
                m_currentMapBounds.mapSpanTheta /= 2;
            else {
                m_currentMapBounds.mapSpanTheta *= 2;
            }
            m_renderHeightmap = true;
            std::cout << "update map span to " << m_currentMapBounds.mapSpanTheta / glm::pi<float>() << "pi" << std::endl;
        }

        // TODO: make sure mapBoundsUniformBuffer doesn't get written during rendering
        memPtr = m_context.device().mapMemory(*m_mapBoundsUniformBuffer.memory, 0, sizeof(m_currentMapBounds));
        std::memcpy(memPtr, &m_currentMapBounds, sizeof(m_currentMapBounds));
        m_context.device().unmapMemory(*m_mapBoundsUniformBuffer.memory);
    }
    vk::SubmitInfo submitInfo{};

    std::vector<vk::Semaphore> mainRenderpassWaitFor = { *m_imageAvailableSemaphores[m_currentFrame] };
    std::vector<vk::PipelineStageFlags> mainRenderpassWaitStage = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    // submit offscreen pass
    if (m_renderHeightmap) {
        submitInfo.waitSemaphoreCount = 0;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &*m_offscreenPassSemaphore;

        vk::CommandBuffer offscreenCommandBuffer = *m_swapchain.noiseCommandBuffers[0];
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &offscreenCommandBuffer;

        m_context.graphicsQueue().submit({ submitInfo }, nullptr);

        mainRenderpassWaitFor.push_back(*m_offscreenPassSemaphore);
        mainRenderpassWaitStage.push_back(vk::PipelineStageFlagBits::eTessellationEvaluationShader);

        m_renderHeightmap = false;
    }

    // execute the command buffer
    submitInfo.waitSemaphoreCount = static_cast<std::uint32_t>(mainRenderpassWaitFor.size());
    submitInfo.pWaitSemaphores = mainRenderpassWaitFor.data();
    submitInfo.pWaitDstStageMask = mainRenderpassWaitStage.data();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*m_swapchain.commandBuffers[imageIndex];

    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*m_renderFinishedSemaphores[m_currentFrame];

    m_context.device().resetFences({ inFlightFence });
    m_context.graphicsQueue().submit({ submitInfo }, inFlightFence);

    // write back to the swap chain
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &*m_renderFinishedSemaphores[m_currentFrame];

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &*m_swapchain.swapchain;
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

void VulkanApplication::run()
{
    using namespace std::chrono;
    m_lastFpsTime = m_lastFrameTime = system_clock::now();
    m_fpsFrameCounter = m_fpsMeasurementsCount = 0;
    m_eyePosition = glm::vec3(0.0f, 3.0f, 0.0f);
    m_lookDirection = glm::vec3(0.0f, -1.0f, 0.0f);
    m_upDirection = glm::vec3(0.0f, 0.0f, 1.0f);
    m_lastCursorPos = glm::vec2(std::numeric_limits<float>::infinity());
    m_currentMapBounds.mapCenterTheta = glm::radians(45.0f);
    m_currentMapBounds.mapCenterPhi = 0;
    m_currentMapBounds.mapSpanTheta = glm::radians(180.0f);

    glfwSetWindowUserPointer(m_context.window(), this);
    glfwSetKeyCallback(m_context.window(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* app = static_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        app->keyEvent(key, scancode, action, mods);
    });

    glfwSetInputMode(m_context.window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(m_context.window(), [](GLFWwindow* window, double xpos, double ypos) {
        auto* app = static_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        if (!std::isfinite(app->m_lastCursorPos.x)) {
            app->m_lastCursorPos = glm::vec2{ xpos, ypos };
        }
        app->m_lastCursorPos += app->m_deltaCursorPos;
        app->m_deltaCursorPos = glm::vec2{ xpos, ypos } - app->m_lastCursorPos;
    });

    glfwSetMouseButtonCallback(m_context.window(), [](GLFWwindow* window, int button, int action, int) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    });

    // main loop
    while (!glfwWindowShouldClose(m_context.window())) {
        drawFrame();

        glfwPollEvents();

        // advance 1 game tick
        step(system_clock::now() - m_lastFrameTime);

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

        // cap framerate
        const double frameInterval = 1.0 / (m_context.refreshRate() + 1);
        while (system_clock::now() - m_lastFrameTime < duration<double>(frameInterval))
            std::this_thread::sleep_for(microseconds(1));
        m_lastFrameTime = currentTime;
    }

    std::cout << "Average fps: " << std::setprecision(0) << std::fixed << m_averageFps << std::endl;
    m_context.device().waitIdle();
}

void VulkanApplication::step(std::chrono::duration<double> delta)
{
    using namespace std::chrono;
    const float dt = duration<float, seconds::period>(delta).count();

    m_planetRotateAngle += dt / 128.0f * glm::radians(90.0f);

    const float sensitivity = 0.005f;
    glm::vec3 right = glm::cross(m_lookDirection, m_upDirection);
    glm::mat4 rotate0 = glm::rotate(glm::mat4(1.0f), -m_deltaCursorPos.x * sensitivity, m_upDirection);
    glm::mat4 rotate1 = glm::rotate(glm::mat4(1.0f), -m_deltaCursorPos.y * sensitivity, right);
    m_lookDirection = rotate0 * rotate1 * glm::vec4(m_lookDirection, 1.0f);
    m_upDirection = rotate1 * glm::vec4(m_upDirection, 1.0f);

    float distance = glm::distance(m_eyePosition, glm::vec3(0)) - 1.0f;
    float speed = distance + 0.1f;
    if (m_movingForward) {
        glm::vec3 velocity = m_lookDirection * speed * dt;
        velocity -= glm::proj(velocity, m_eyePosition) * glm::clamp(1.0f - 5.0f * distance, 0.0f, 1.0f);
        m_eyePosition += velocity;
    }
    if (m_movingBackward) {
        m_eyePosition -= m_lookDirection * dt;
    }
    if (m_rotatingLeft) {
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), -dt * 1.5f, m_lookDirection);
        m_upDirection = rotate * glm::vec4(m_upDirection, 0.0f);
    }
    if (m_rotatingRight) {
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), dt * 1.5f, m_lookDirection);
        m_upDirection = rotate * glm::vec4(m_upDirection, 0.0f);
    }

    m_lastCursorPos += m_deltaCursorPos;
    m_deltaCursorPos = glm::vec2(0);
}

void VulkanApplication::keyEvent(int key, int, int action, int)
{
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        m_context.toggleFullscreenMode();
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetInputMode(m_context.window(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_lastCursorPos = glm::vec2(std::numeric_limits<float>::infinity());
    }

    if (key == GLFW_KEY_W) {
        if (action == GLFW_PRESS) {
            m_movingForward = true;
        } else if (action == GLFW_RELEASE) {
            m_movingForward = false;
        }
    }
    if (key == GLFW_KEY_S) {
        if (action == GLFW_PRESS) {
            m_movingBackward = true;
        } else if (action == GLFW_RELEASE) {
            m_movingBackward = false;
        }
    }
    if (key == GLFW_KEY_A) {
        if (action == GLFW_PRESS) {
            m_rotatingLeft = true;
        } else if (action == GLFW_RELEASE) {
            m_rotatingLeft = false;
        }
    }
    if (key == GLFW_KEY_D) {
        if (action == GLFW_PRESS) {
            m_rotatingRight = true;
        } else if (action == GLFW_RELEASE) {
            m_rotatingRight = false;
        }
    }
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
