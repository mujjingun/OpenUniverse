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
static const std::uint64_t timeOut = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

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
    glm::mat4 iMVP;
    glm::mat4 shadowVP;
    glm::vec4 eyePos;
    glm::vec4 modelEyePos;
    glm::vec4 lightPos;
    std::int32_t parallelCount;
    std::int32_t meridianCount;
    std::uint32_t readyNoiseImageIndex;
};

VulkanApplication::VulkanApplication()
    : m_context(600, 480, false)

    // figure out swapchain properties
    , m_swapchainProps(m_context.selectSwapchainProperties())

    // make swapchain
    , m_swapchain(m_context, m_swapchainProps)

    // make semaphores for synchronizing frame drawing operations
    , m_imageAvailableSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_renderFinishedSemaphores(m_context.makeSemaphores(maxFramesInFlight))
    , m_inFlightFences(m_context.makeFences(maxFramesInFlight, true))

    // make textures
    , m_textureImage(m_context.makeTextureImage("textures/texture.jpg"))

    // make sampler
    , m_sampler(m_context.makeTextureSampler(false))

    , m_parallelCount(100)
    , m_meridianCount(100)
    , m_mapBounds(m_swapchain.noiseImages.size())
{
    // make uniform buffers
    for (std::uint32_t i = 0; i < m_swapchainProps.imageCount; ++i) {
        m_uniformBuffers.push_back(
            m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UniformBufferObject)));
        m_renderMapBoundsUniformBuffers.push_back(
            m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MapBoundsObject)));
        m_shadowInfoBuffers.push_back(
            m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UniformBufferObject)));
    }
    for (std::size_t i = 0; i < m_swapchain.noiseImages.size(); ++i) {
        m_mapBoundsUniformBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MapBoundsObject)));
    }
    for (int i = 0; i < 3; ++i) {
        m_numberBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(std::uint32_t)));
    }

    // record draw commands
    recordDrawCommands();
}

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();

    m_updateOverallmap = true;
    m_renderingHeightmap = false;
    m_updateHeightmap = true;

    // recreate swapchain
    m_swapchainProps = m_context.selectSwapchainProperties();
    m_swapchain = SwapchainObject(m_context, m_swapchainProps, *m_swapchain.swapchain);

    // re-record draw commands
    recordDrawCommands();
}

static std::uint32_t ceilDiv(std::uint32_t x, std::uint32_t y)
{
    return 1 + ((x - 1) / y);
}

void VulkanApplication::recordDrawCommands()
{
    vk::ClearColorValue clearColor(std::array<float, 4>{ { 0.0f, 0.0f, 0.0f, 1.0f } });
    vk::ClearDepthStencilValue clearDepth(1.0f, 0.0f);

    // record offscreen rendering
    {
        std::size_t index = 0;
        for (vk::UniqueCommandBuffer const& commandBuffer : m_swapchain.noiseCommandBuffers) {
            ImageObject const& image = m_swapchain.noiseImages[index];

            vk::CommandBufferBeginInfo beginInfo{};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
            beginInfo.pInheritanceInfo = nullptr;

            commandBuffer->begin(beginInfo);

            // bind uniform descriptor sets
            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = *m_mapBoundsUniformBuffers[index].buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites[0].dstSet = m_swapchain.noiseDescriptorSet.sets[index];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = vk::ImageLayout::eGeneral;
            imageInfo.imageView = *image.view;
            imageInfo.sampler = nullptr;

            descriptorWrites[1].dstSet = m_swapchain.noiseDescriptorSet.sets[index];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eStorageImage;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            m_context.device().updateDescriptorSets(descriptorWrites, {});

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *m_swapchain.noisePipeline);

            commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_swapchain.noisePipelineLayout, 0,
                1, &m_swapchain.noiseDescriptorSet.sets[index], 0, nullptr);

            commandBuffer->dispatch(ceilDiv(image.extent.width, 32), ceilDiv(image.extent.height, 32), 1);

            vk::BufferImageCopy region{};
            region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            region.imageSubresource.layerCount = image.layerCount;

            region.imageExtent.width = image.extent.width;
            region.imageExtent.height = image.extent.height;
            region.imageExtent.depth = 1;

            commandBuffer->copyImageToBuffer(*image.image, vk::ImageLayout::eTransferSrcOptimal, *m_swapchain.terrain.buffer, { region });

            commandBuffer->end();

            index++;
        }
    }

    // record main rendering
    {
        for (std::size_t index = 0; index < m_swapchainProps.imageCount; ++index) {

            // bind uniform descriptor sets
            std::vector<vk::WriteDescriptorSet> descriptorWrites{};

            // shadow planet info
            vk::DescriptorBufferInfo shadowPlanetInfo{};
            shadowPlanetInfo.buffer = *m_shadowInfoBuffers[index].buffer;
            shadowPlanetInfo.offset = 0;
            shadowPlanetInfo.range = VK_WHOLE_SIZE;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.shadowPlanetDescriptorSet.sets[index], 0, 0,
                1, vk::DescriptorType::eUniformBuffer, nullptr, &shadowPlanetInfo));

            // planet info
            vk::DescriptorBufferInfo planetInfo{};
            planetInfo.buffer = *m_uniformBuffers[index].buffer;
            planetInfo.offset = 0;
            planetInfo.range = VK_WHOLE_SIZE;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.planetDescriptorSet.sets[index], 0, 0,
                1, vk::DescriptorType::eUniformBuffer, nullptr, &planetInfo));

            // map bounds info
            vk::DescriptorBufferInfo mapBoundsUniformBufferInfo{};
            mapBoundsUniformBufferInfo.buffer = *m_renderMapBoundsUniformBuffers[index].buffer;
            mapBoundsUniformBufferInfo.offset = 0;
            mapBoundsUniformBufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.planetDescriptorSet.sets[index], 1, 0,
                1, vk::DescriptorType::eUniformBuffer, nullptr, &mapBoundsUniformBufferInfo));

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.shadowPlanetDescriptorSet.sets[index], 1, 0,
                1, vk::DescriptorType::eUniformBuffer, nullptr, &mapBoundsUniformBufferInfo));

            // map images
            std::vector<vk::DescriptorImageInfo> noiseImageInfos{};
            for (auto const& image : m_swapchain.noiseImages) {
                vk::DescriptorImageInfo imageInfo;
                imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                imageInfo.imageView = *image.view;
                imageInfo.sampler = *m_sampler;
                noiseImageInfos.push_back(imageInfo);
            }

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.planetDescriptorSet.sets[index], 2, 0,
                static_cast<std::uint32_t>(noiseImageInfos.size()),
                vk::DescriptorType::eCombinedImageSampler, noiseImageInfos.data()));

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.shadowPlanetDescriptorSet.sets[index], 2, 0,
                static_cast<std::uint32_t>(noiseImageInfos.size()),
                vk::DescriptorType::eCombinedImageSampler, noiseImageInfos.data()));

            // shadow image
            vk::DescriptorImageInfo shadowMapImageInfo{};
            shadowMapImageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            shadowMapImageInfo.imageView = *m_swapchain.shadowImages[index].view;
            shadowMapImageInfo.sampler = *m_sampler;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.shadowMapDescriptorSet.sets[index], 0, 0,
                1, vk::DescriptorType::eCombinedImageSampler, &shadowMapImageInfo));

            // fps indicator
            std::vector<vk::DescriptorBufferInfo> numBufInfos{};
            for (auto const& buf : m_numberBuffers) {
                numBufInfos.push_back(vk::DescriptorBufferInfo(*buf.buffer, 0, VK_WHOLE_SIZE));
            }

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.numbersDescriptorSet.sets[index], 0, 0,
                static_cast<std::uint32_t>(numBufInfos.size()),
                vk::DescriptorType::eUniformBuffer, nullptr, numBufInfos.data()));

            // hdr image for downsampling
            vk::DescriptorImageInfo brightHdrImageInfo{};
            brightHdrImageInfo.imageLayout = vk::ImageLayout::eGeneral;
            brightHdrImageInfo.imageView = *m_swapchain.hdrImages[index].view;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.downsampleDescriptorSet.sets[index], 0, 0,
                1, vk::DescriptorType::eStorageImage, &brightHdrImageInfo));

            // bright passed image for bloom input
            vk::DescriptorImageInfo brightPassImageInfo{};
            brightPassImageInfo.imageLayout = vk::ImageLayout::eGeneral;
            brightPassImageInfo.imageView = *m_swapchain.brightImages[index].view;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.bloomDescriptorSet.sets[index], 0, 0, // bloomh writes to this, bloomv reads this
                1, vk::DescriptorType::eStorageImage, &brightPassImageInfo));

            // bloom result image
            vk::DescriptorImageInfo bloomResultImageInfo{};
            bloomResultImageInfo.imageLayout = vk::ImageLayout::eGeneral;
            bloomResultImageInfo.imageView = *m_swapchain.bloomImages[index].view;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.downsampleDescriptorSet.sets[index], 1, 0,
                1, vk::DescriptorType::eStorageImage, &bloomResultImageInfo));

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.bloomDescriptorSet.sets[index], 1, 0, // bloomh reads this, bloomv writes to this
                1, vk::DescriptorType::eStorageImage, &bloomResultImageInfo));

            // bloom image for composite input
            vk::DescriptorImageInfo bloomImageInfo{};
            bloomImageInfo.imageLayout = vk::ImageLayout::eGeneral;
            bloomImageInfo.imageView = *m_swapchain.bloomImages[index].view;
            bloomImageInfo.sampler = *m_sampler;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.sceneDescriptorSet.sets[index], 0, 0,
                1, vk::DescriptorType::eCombinedImageSampler, &bloomImageInfo));

            // hdr image for composite input
            vk::DescriptorImageInfo hdrImageInfo{};
            hdrImageInfo.imageLayout = vk::ImageLayout::eGeneral;
            hdrImageInfo.imageView = *m_swapchain.hdrImages[index].view;
            hdrImageInfo.sampler = *m_sampler;

            descriptorWrites.push_back(vk::WriteDescriptorSet(
                m_swapchain.sceneDescriptorSet.sets[index], 1, 0,
                1, vk::DescriptorType::eCombinedImageSampler, &hdrImageInfo));

            m_context.device().updateDescriptorSets(descriptorWrites, {});

            // begin recording
            vk::CommandBufferBeginInfo beginInfo{};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
            beginInfo.pInheritanceInfo = nullptr;

            vk::CommandBuffer commandBuffer = *m_swapchain.commandBuffers[index];
            commandBuffer.begin(beginInfo);

            auto planetVertexCount = static_cast<std::uint32_t>(m_parallelCount * m_meridianCount * 4);

            // render shadow depth
            {
                std::array<vk::ClearValue, 1> clearValues = { { clearDepth } };

                vk::RenderPassBeginInfo shadowRenderPassInfo{};
                shadowRenderPassInfo.renderPass = *m_swapchain.shadowRenderPass;
                shadowRenderPassInfo.framebuffer = *m_swapchain.shadowFramebuffers[index];
                shadowRenderPassInfo.renderArea.offset.x = 0;
                shadowRenderPassInfo.renderArea.offset.y = 0;
                shadowRenderPassInfo.renderArea.extent = m_swapchainProps.extent;
                shadowRenderPassInfo.clearValueCount = clearValues.size();
                shadowRenderPassInfo.pClearValues = clearValues.data();

                commandBuffer.beginRenderPass(shadowRenderPassInfo, vk::SubpassContents::eInline);

                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.shadowPlanetPipelineLayout, 0,
                    { m_swapchain.shadowPlanetDescriptorSet.sets[index] }, {});

                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.shadowPlanetPipeline);
                commandBuffer.draw(planetVertexCount, 1, 0, 0);

                commandBuffer.endRenderPass();
            }

            // main render pass
            {
                std::array<vk::ClearValue, 2> clearValues = { { clearColor, clearDepth } };

                vk::RenderPassBeginInfo mainRenderPassInfo{};
                mainRenderPassInfo.renderPass = *m_swapchain.hdrRenderPass;
                mainRenderPassInfo.framebuffer = *m_swapchain.framebuffers[index];
                mainRenderPassInfo.renderArea.offset.x = 0;
                mainRenderPassInfo.renderArea.offset.y = 0;
                mainRenderPassInfo.renderArea.extent = m_swapchainProps.extent;
                mainRenderPassInfo.clearValueCount = clearValues.size();
                mainRenderPassInfo.pClearValues = clearValues.data();

                commandBuffer.beginRenderPass(mainRenderPassInfo, vk::SubpassContents::eInline);

                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.planetPipelineLayout, 0,
                    { m_swapchain.planetDescriptorSet.sets[index], m_swapchain.shadowMapDescriptorSet.sets[index] }, {});

                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.planetPipeline);
                commandBuffer.draw(planetVertexCount, 1, 0, 0);

                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.atmospherePipeline);
                commandBuffer.draw(3, 1, 0, 0);

                commandBuffer.endRenderPass();
            }

            // bloom
            {
                vk::Extent2D extent = m_swapchain.brightImages[index].extent;

                // clear images
                transitionImageLayout(commandBuffer, *m_swapchain.brightImages[index].image, 1,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
                commandBuffer.clearColorImage(*m_swapchain.brightImages[index].image, vk::ImageLayout::eTransferDstOptimal,
                    { clearColor }, { vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) });
                transitionImageLayout(commandBuffer, *m_swapchain.brightImages[index].image, 1,
                    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, 1);

                transitionImageLayout(commandBuffer, *m_swapchain.bloomImages[index].image, 1,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
                commandBuffer.clearColorImage(*m_swapchain.bloomImages[index].image, vk::ImageLayout::eTransferDstOptimal,
                    { clearColor }, { vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) });
                transitionImageLayout(commandBuffer, *m_swapchain.bloomImages[index].image, 1,
                    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, 1);

                // downsample
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_swapchain.downsamplePipelineLayout, 0,
                    1, &m_swapchain.downsampleDescriptorSet.sets[index], 0, nullptr);
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_swapchain.downsamplePipeline);
                commandBuffer.dispatch(ceilDiv(extent.width / 2, 32), ceilDiv(extent.height, 32), 1);

                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_swapchain.bloomPipelineLayout, 0,
                    1, &m_swapchain.bloomDescriptorSet.sets[index], 0, nullptr);

                // bloom horizontally
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_swapchain.bloomHPipeline);
                commandBuffer.dispatch(ceilDiv(extent.width, 32), ceilDiv(extent.height, 32), 1);

                // bloom vertically
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_swapchain.bloomVPipeline);
                commandBuffer.dispatch(ceilDiv(extent.width, 32), ceilDiv(extent.height, 32), 1);
            }

            // bloom vertically + combine with hdr
            {
                std::array<vk::ClearValue, 1> clearValues = { { clearColor } };

                vk::RenderPassBeginInfo presentRenderPassInfo{};
                presentRenderPassInfo.renderPass = *m_swapchain.presentRenderPass;
                presentRenderPassInfo.framebuffer = *m_swapchain.presentFramebuffers[index];
                presentRenderPassInfo.renderArea.offset.x = 0;
                presentRenderPassInfo.renderArea.offset.y = 0;
                presentRenderPassInfo.renderArea.extent = m_swapchainProps.extent;
                presentRenderPassInfo.clearValueCount = clearValues.size();
                presentRenderPassInfo.pClearValues = clearValues.data();

                commandBuffer.beginRenderPass(presentRenderPassInfo, vk::SubpassContents::eInline);

                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.scenePipelineLayout, 0,
                    { m_swapchain.sceneDescriptorSet.sets[index] }, {});
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.scenePipeline);
                commandBuffer.draw(3, 1, 0, 0);

                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.numbersPipelineLayout, 0,
                    { m_swapchain.numbersDescriptorSet.sets[index] }, {});
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.numbersPipeline);
                commandBuffer.draw(3, 1, 0, 0);

                commandBuffer.endRenderPass();
            }

            commandBuffer.end();
        }
    }
}

void VulkanApplication::drawFrame()
{
    // The drawFrame function will perform the following operations:
    // 1. Acquire an image from the swap chain
    // 2. Execute the command buffer with that image as attachment in the framebuffer
    // 3. Return the image to the swap chain for presentation

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
        ubo.model = glm::rotate(glm::mat4(1.0f), m_planetRotateAngle, glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f)));
        ubo.model = glm::scale(ubo.model, glm::vec3(1.0f, 1.0f, 1.0f));

        ubo.eyePos = glm::vec4(m_eyePosition, 0);
        ubo.view = glm::lookAt(m_eyePosition, m_eyePosition + m_lookDirection, m_upDirection);

        ubo.modelEyePos = glm::inverse(ubo.model) * ubo.eyePos;

        // TODO: try logarithmic depth?
        float near = 0.5f * (length(glm::vec3(ubo.modelEyePos)) - 1.0f);
        ubo.proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(m_swapchainProps.extent.width) / m_swapchainProps.extent.height,
            near, near * 100.0f);
        ubo.proj[1][1] *= -1; // invert Y axis

        ubo.iMVP = glm::inverse(ubo.proj * ubo.view * ubo.model);

        ubo.lightPos = glm::vec4(glm::vec3(200, -200, 0), 0);

        glm::vec3 worldPlanetCenter = glm::vec3(ubo.model * glm::vec4(0));
        float planetToLight = glm::distance(worldPlanetCenter, glm::vec3(ubo.lightPos));

        // TODO: adjust this
        glm::mat4 shadowView = glm::lookAt(glm::vec3(ubo.lightPos), worldPlanetCenter, glm::vec3(0, 0, 1));
        glm::mat4 shadowProj = glm::perspective(glm::radians(0.5f),
            static_cast<float>(m_swapchain.shadowImages[imageIndex].extent.width) / m_swapchain.shadowImages[imageIndex].extent.height,
            planetToLight - 1.1f, planetToLight + 1.1f);
        shadowProj[1][1] *= -1; // invert Y axis

        ubo.shadowVP = shadowProj * shadowView;

        ubo.parallelCount = m_parallelCount;
        ubo.meridianCount = m_meridianCount;

        std::size_t nextOffscreenIndex = m_lastRenderedIndex + 1;
        if (nextOffscreenIndex >= m_swapchain.noiseImages.size()) {
            nextOffscreenIndex = 1;
        }
        if (m_renderingHeightmap && m_context.device().getFenceStatus(*m_swapchain.noiseFences[nextOffscreenIndex]) == vk::Result::eSuccess) {
            m_context.device().resetFences({ *m_swapchain.noiseFences[nextOffscreenIndex] });
            m_lastRenderedIndex = nextOffscreenIndex;
            m_renderingHeightmap = false;

            std::cout << "finished. current index: " << m_lastRenderedIndex << std::endl;
        }
        ubo.readyNoiseImageIndex = static_cast<std::uint32_t>(m_lastRenderedIndex);

        m_context.updateMemory(*m_uniformBuffers[imageIndex].memory, &ubo, sizeof(UniformBufferObject));
        m_context.updateMemory(*m_renderMapBoundsUniformBuffers[imageIndex].memory, &m_mapBounds[m_lastRenderedIndex], sizeof(MapBoundsObject));

        UniformBufferObject shadowUbo = ubo;
        shadowUbo.eyePos = ubo.lightPos;
        shadowUbo.view = shadowView;
        shadowUbo.proj = shadowProj;

        m_context.updateMemory(*m_shadowInfoBuffers[imageIndex].memory, &shadowUbo, sizeof(UniformBufferObject));

        for (std::uint32_t p = 100, i = 0; p > 0; p /= 10, ++i) {
            std::uint32_t digit = m_currentFps / p % 10;
            m_context.updateMemory(*m_numberBuffers[i].memory, &digit, sizeof(std::uint32_t));
        }

        // update map bounds
        if (!m_renderingHeightmap) {
            std::size_t nextOffscreenIndex = m_lastRenderedIndex + 1;
            if (nextOffscreenIndex >= m_swapchain.noiseImages.size()) {
                nextOffscreenIndex = 1;
            }

            const MapBoundsObject mapBounds = m_mapBounds[m_lastRenderedIndex];
            const glm::vec3 oldMapCenter = {
                std::sin(mapBounds.mapCenterTheta) * std::cos(mapBounds.mapCenterPhi),
                std::sin(mapBounds.mapCenterTheta) * std::sin(mapBounds.mapCenterPhi),
                std::cos(mapBounds.mapCenterTheta)
            };

            const glm::vec3 normEye = glm::normalize(ubo.modelEyePos);
            const float angle = std::acos(glm::dot(oldMapCenter, normEye));
            const float newMapSpan = std::acos(1.0f / length(ubo.modelEyePos));
            const bool tooLarge = newMapSpan < mapBounds.mapSpanTheta / 2;
            const bool tooSmall = newMapSpan > mapBounds.mapSpanTheta;
            const bool outOfRange = angle > newMapSpan;

            MapBoundsObject newMapBounds;
            if (m_updateHeightmap || tooLarge || tooSmall || outOfRange) {
                newMapBounds.mapCenterTheta = std::acos(normEye.z);
                newMapBounds.mapCenterPhi = std::atan2(normEye.y, normEye.x);

                if (tooLarge)
                    newMapBounds.mapSpanTheta = mapBounds.mapSpanTheta / 2;
                else if (tooSmall) {
                    newMapBounds.mapSpanTheta = mapBounds.mapSpanTheta * 2;
                } else {
                    newMapBounds.mapSpanTheta = mapBounds.mapSpanTheta;
                }

                m_updateHeightmap = false;
                m_mapBounds[nextOffscreenIndex] = newMapBounds;

                m_context.updateMemory(
                    *m_mapBoundsUniformBuffers[nextOffscreenIndex].memory, &m_mapBounds[nextOffscreenIndex], sizeof(MapBoundsObject));

                // submit compute shader redraw
                vk::SubmitInfo submitInfo{};
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &*m_swapchain.noiseCommandBuffers[nextOffscreenIndex];

                m_context.computeQueue().submit({ submitInfo }, *m_swapchain.noiseFences[nextOffscreenIndex]);

                m_renderingHeightmap = true;

                std::cout << "updating map span to " << newMapBounds.mapSpanTheta / glm::pi<float>() << "pi, "
                          << "theta = " << newMapBounds.mapCenterTheta
                          << ", phi = " << newMapBounds.mapCenterPhi << std::endl;
            }
        }
    }

    vk::SubmitInfo submitInfo{};

    // submit offscreen pass
    if (m_updateOverallmap) {
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*m_swapchain.noiseCommandBuffers[0];

        m_context.computeQueue().submit({ submitInfo }, *m_swapchain.noiseFences[0]);

        m_updateOverallmap = false;

        std::cout << "updating overall map" << std::endl;
    }

    // execute the command buffer
    std::vector<vk::Semaphore> mainRenderpassWaitFor = { *m_imageAvailableSemaphores[m_currentFrame] };
    std::vector<vk::PipelineStageFlags> mainRenderpassWaitStage = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

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

    m_planetRotateAngle = 0.0f;

    m_eyePosition = glm::vec3(0.0f, -3.0f, 0.0f);
    m_lookDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    m_upDirection = glm::vec3(0.0f, 0.0f, 1.0f);
    m_movingForward = m_movingBackward = m_rotatingLeft = m_rotatingRight = false;

    m_lastCursorPos = glm::vec2(std::numeric_limits<float>::infinity());

    m_updateOverallmap = true;
    m_updateHeightmap = true;
    m_renderingHeightmap = false;
    m_lastRenderedIndex = 0;
    m_mapBounds[0].mapCenterTheta = 0;
    m_mapBounds[0].mapCenterPhi = 0;
    m_mapBounds[0].mapSpanTheta = glm::radians(180.0f);
    m_context.updateMemory(*m_mapBoundsUniformBuffers[0].memory, &m_mapBounds[0], sizeof(MapBoundsObject));

    glfwSetWindowUserPointer(m_context.window(), this);
    glfwSetKeyCallback(m_context.window(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* app = static_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        app->keyEvent(key, scancode, action, mods);
    });

    //glfwSetInputMode(m_context.window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(m_context.window(), [](GLFWwindow* window, double xpos, double ypos) {
        auto* app = static_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        if (!std::isfinite(app->m_lastCursorPos.x)) {
            app->m_lastCursorPos = glm::vec2{ xpos, ypos };
        }
        app->m_deltaCursorPos = glm::vec2{ xpos, ypos } - app->m_lastCursorPos;
    });

    glfwSetMouseButtonCallback(m_context.window(), [](GLFWwindow* window, int button, int action, int) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    });

    // main loop
    while (!glfwWindowShouldClose(m_context.window())) {
        glfwPollEvents();

        // advance 1 game tick
        const auto currentTime = system_clock::now();
        const auto delta = currentTime - m_lastFrameTime;
        m_lastFrameTime = currentTime;
        step(delta);

        // draw to screen
        drawFrame();

        // calculate FPS
        m_fpsFrameCounter++;
        const auto elapsedTime = currentTime - m_lastFpsTime;
        if (elapsedTime >= seconds(1)) {
            double fps = static_cast<double>(m_fpsFrameCounter) / duration_cast<seconds>(elapsedTime).count();
            m_currentFps = static_cast<std::uint32_t>(std::round(fps));

            m_lastFpsTime = currentTime;
            m_fpsFrameCounter = 0;
            m_fpsMeasurementsCount++;
        }

        // cap framerate
        //const double frameInterval = 1.0 / (m_context.refreshRate() + 1);
        //while (system_clock::now() - m_lastFrameTime < duration<double>(frameInterval))
        //    std::this_thread::sleep_for(microseconds(1));
    }

    // TODO: temp workaround for freezing in fullscreen mode
    if (m_context.isFullscreen()) {
        m_context.toggleFullscreenMode();
    }
    m_context.device().waitIdle();
}

void VulkanApplication::step(std::chrono::duration<double> delta)
{
    using namespace std::chrono;
    const float dt = duration<float, seconds::period>(delta).count();

    m_planetRotateAngle += dt / 16.0f * glm::radians(90.0f);

    const float r = 1 - std::exp(-dt * 10.0f);
    glm::vec2 smoothDelta{};
    if (std::isfinite(m_lastCursorPos.x)) {
        smoothDelta = r * m_deltaCursorPos;
    }

    const float sensitivity = 0.005f;
    glm::vec3 right = glm::cross(m_lookDirection, m_upDirection);
    glm::mat4 rotate0 = glm::rotate(glm::mat4(1.0f), -smoothDelta.x * sensitivity, m_upDirection);
    glm::mat4 rotate1 = glm::rotate(glm::mat4(1.0f), -smoothDelta.y * sensitivity, right);
    m_lookDirection = rotate0 * rotate1 * glm::vec4(m_lookDirection, 1.0f);
    m_upDirection = rotate1 * glm::vec4(m_upDirection, 1.0f);

    float distance = glm::distance(m_eyePosition, glm::vec3(0)) - 1.0f;
    float speed = distance + 0.02f;
    if (m_movingForward) {
        glm::vec3 velocity = m_lookDirection * speed * dt;
        velocity -= glm::proj(velocity, m_eyePosition) * glm::clamp(1.0f - 100.0f * distance, 0.0f, 1.0f);
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

    m_lastCursorPos += smoothDelta;
    m_deltaCursorPos -= smoothDelta;
}

void VulkanApplication::keyEvent(int key, int, int action, int)
{
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        m_context.toggleFullscreenMode();
        m_lastCursorPos = glm::vec2(std::numeric_limits<float>::infinity());
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
        try {
            app.run();
        } catch (std::exception const& err) {
            std::cerr << "Exception raised while running the program: " << err.what() << std::endl;
            std::terminate();
        }
    } catch (std::exception const& err) {
        std::cerr << "Exception raised while setting up the program: " << err.what() << std::endl;
        std::terminate();
    }

    return 0;
}
