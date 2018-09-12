#include "main.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/intersect.hpp"
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

struct MapBoundsObject {
};

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

    // make offscreen render target
    noiseImage = context.makeImage(vk::SampleCountFlagBits::e1, 1, properties.extent, vk::Format::eR32G32B32A32Sfloat,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::ImageAspectFlagBits::eColor);
    noiseMultiSampleImage = context.makeMultiSampleImage(noiseImage.format, properties.extent, sampleCount);

    // make render pass
    renderPass = context.makeRenderPass(sampleCount, properties.surfaceFormat.format, depthImage.format, 2);
    noiseRenderPass = context.makeRenderPass(sampleCount, noiseImage.format, depthImage.format, 1);

    // make pipelines
    pipelineLayout = context.makePipelineLayout(descriptorSetLayout);
    terrainPipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 0, sampleCount,
        "shaders/planet.vert.spv", "shaders/planet.frag.spv", "shaders/planet.tesc.spv", "shaders/planet.tese.spv",
        vk::PrimitiveTopology::ePatchList, false, {}, {});

    atmospherePipeline = context.makePipeline(*pipelineLayout, properties.extent, *renderPass, 1, sampleCount,
        "shaders/air.vert.spv", "shaders/air.frag.spv", "shaders/air.tesc.spv", "shaders/air.tese.spv",
        vk::PrimitiveTopology::ePatchList, false, {}, {});

    noisePipeline = context.makePipeline(*pipelineLayout, properties.extent, *noiseRenderPass, 0, sampleCount,
        "shaders/noise.vert.spv", "shaders/noise.frag.spv", nullptr, nullptr,
        vk::PrimitiveTopology::eTriangleFan, false, {}, {});

    // make command buffers
    commandBuffers = context.allocateCommandBuffers(swapchainImageCount);
    noiseCommandBuffer = std::move(context.allocateCommandBuffers(1)[0]);

    // make framebuffers
    for (auto const& swapchainImageView : swapchainImageViews) {
        framebuffers.push_back(context.makeFramebuffer(*swapchainImageView, *depthImage.view,
            *multiSampleImage.view, *renderPass, properties.extent));
    }
    noiseFramebuffer = context.makeFramebuffer(*noiseImage.view, *depthImage.view,
        *noiseMultiSampleImage.view, *noiseRenderPass, properties.extent);
}

VulkanApplication::VulkanApplication()
    : m_context(600, 480, false)

    // figure out swapchain properties
    , m_swapchainProps(m_context.selectSwapchainProperties())

    // make description set layout, pool, and sets
    // TODO: make separate descriptionsetlayout for noise rendering
    , m_descriptorSetLayout(m_context.makeDescriptorSetLayout())
    , m_descriptorPool(m_context.makeDescriptorPool(m_swapchainProps.imageCount + 1))
    , m_descriptorSets(m_context.makeDescriptorSets(*m_descriptorPool, *m_descriptorSetLayout, m_swapchainProps.imageCount + 1))

    // make swapchain
    , m_swapchain(m_context, *m_descriptorSetLayout, m_swapchainProps)

    // make semaphores for synchronizing frame drawing operations
    , m_imageAvailableSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_renderFinishedSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_offscreenPassSemaphore(std::move(m_context.makeSemaphores(1)[0]))
    , m_inFlightFences(m_context.makeFences(maxFramesInFlight))

    // make textures
    , m_textureImage(m_context.makeTextureImage("textures/texture.jpg"))

    // make sampler
    , m_sampler(m_context.makeTextureSampler(false))

    , m_parallelCount(45)
    , m_meridianCount(90)
    , m_renderHeightmap(true)
{
    // make uniform buffers
    for (std::uint32_t i = 0; i < m_swapchainProps.imageCount; ++i) {
        m_uniformBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UniformBufferObject)));
    }
    m_noiseUniformBuffer = m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MapBoundsObject));

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();
    m_renderHeightmap = true;

    // recreate swapchain
    m_swapchainProps = m_context.selectSwapchainProperties();
    m_swapchain = SwapchainObject(m_context, *m_descriptorSetLayout, m_swapchainProps, *m_swapchain.swapchain);

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
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        beginInfo.pInheritanceInfo = nullptr;

        m_swapchain.noiseCommandBuffer->begin(beginInfo);

        // bind uniform descriptor sets
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *m_noiseUniformBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};
        descriptorWrites[0].dstSet = m_descriptorSets.back();
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        m_context.device().updateDescriptorSets(descriptorWrites, {});

        vk::RenderPassBeginInfo noiseRenderPassInfo{};
        noiseRenderPassInfo.renderPass = *m_swapchain.noiseRenderPass;
        noiseRenderPassInfo.framebuffer = *m_swapchain.noiseFramebuffer;
        noiseRenderPassInfo.renderArea.offset.x = 0;
        noiseRenderPassInfo.renderArea.offset.y = 0;
        noiseRenderPassInfo.renderArea.extent = m_swapchainProps.extent;
        noiseRenderPassInfo.clearValueCount = clearValues.size();
        noiseRenderPassInfo.pClearValues = clearValues.data();

        m_swapchain.noiseCommandBuffer->beginRenderPass(noiseRenderPassInfo, vk::SubpassContents::eInline);
        {
            m_swapchain.noiseCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.pipelineLayout, 0,
                { m_descriptorSets.back() }, {});

            m_swapchain.noiseCommandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.noisePipeline);
            m_swapchain.noiseCommandBuffer->draw(4, 1, 0, 0);
        }
        m_swapchain.noiseCommandBuffer->endRenderPass();

        m_swapchain.noiseCommandBuffer->end();
    }

    // record main rendering
    std::size_t index = 0;
    for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.commandBuffers) {

        // bind uniform descriptor sets
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *m_uniformBuffers[index].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].dstSet = m_descriptorSets[index];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = *m_swapchain.noiseImage.view;
        imageInfo.sampler = *m_sampler;

        descriptorWrites[1].dstSet = m_descriptorSets[index];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

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
                { m_descriptorSets[index] }, {});

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

void VulkanApplication::run()
{
    using namespace std::chrono;
    m_lastFpsTime = m_lastFrameTime = system_clock::now();
    m_fpsFrameCounter = m_fpsMeasurementsCount = 0;
    m_eyePosition = glm::vec3(0.0f, 3.0f, 0.0f);
    m_lookDirection = glm::vec3(0.0f, -1.0f, 0.0f);
    m_upDirection = glm::vec3(0.0f, 0.0f, 1.0f);
    m_lastCursorPos = glm::vec2(std::numeric_limits<float>::infinity());

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

    m_planetRotateAngle += dt / 32.0f * glm::radians(90.0f);

    const float sensitivity = 0.005f;
    glm::vec3 right = glm::cross(m_lookDirection, m_upDirection);
    glm::mat4 rotate0 = glm::rotate(glm::mat4(1.0f), m_deltaCursorPos.x * sensitivity, m_upDirection);
    glm::mat4 rotate1 = glm::rotate(glm::mat4(1.0f), m_deltaCursorPos.y * sensitivity, right);
    m_lookDirection = rotate0 * rotate1 * glm::vec4(m_lookDirection, 1.0f);
    m_upDirection = rotate1 * glm::vec4(m_upDirection, 1.0f);

    float distance = glm::distance(m_eyePosition, glm::vec3(0));
    float speed = (distance - 1.0f);
    if (m_movingForward) {
        m_eyePosition += m_lookDirection * speed * dt;
    }
    if (m_movingBackward) {
        m_eyePosition -= m_lookDirection * speed * dt;
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
        MapBoundsObject mapBounds;
        void* const data = m_context.device().mapMemory(*m_noiseUniformBuffer.memory, 0, sizeof(mapBounds));
        std::memcpy(data, &mapBounds, sizeof(mapBounds));
        m_context.device().unmapMemory(*m_noiseUniformBuffer.memory);
    }
    vk::SubmitInfo submitInfo{};

    vk::Semaphore mainRenderpassWaitFor = *m_imageAvailableSemaphores[m_currentFrame];
    vk::PipelineStageFlags mainRenderpassWaitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    // submit offscreen pass
    if (m_renderHeightmap) {
        std::array<vk::PipelineStageFlags, 1> noiseWaitStages = {
            { vk::PipelineStageFlagBits::eColorAttachmentOutput }
        };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &*m_imageAvailableSemaphores[m_currentFrame];
        submitInfo.pWaitDstStageMask = noiseWaitStages.data();

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &*m_offscreenPassSemaphore;

        vk::CommandBuffer offscreenCommandBuffer = *m_swapchain.noiseCommandBuffer;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &offscreenCommandBuffer;

        m_context.graphicsQueue().submit({ submitInfo }, nullptr);

        mainRenderpassWaitFor = *m_offscreenPassSemaphore;
        mainRenderpassWaitStage = vk::PipelineStageFlagBits::eTessellationEvaluationShader;

        m_renderHeightmap = false;
    }

    // update the uniform data
    {
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

        // update uniform buffer
        void* const data = m_context.device().mapMemory(*m_uniformBuffers[imageIndex].memory, 0, sizeof(ubo));
        std::memcpy(data, &ubo, sizeof(ubo));
        m_context.device().unmapMemory(*m_uniformBuffers[imageIndex].memory);
    }

    // execute the command buffer
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &mainRenderpassWaitFor;
    submitInfo.pWaitDstStageMask = &mainRenderpassWaitStage;

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
