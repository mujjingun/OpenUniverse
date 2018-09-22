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
    glm::mat4 iMVP;
    glm::vec4 eyePos;
    glm::vec4 modelEyePos;
    glm::vec4 lightDir;
    std::int32_t parallelCount;
    std::int32_t meridianCount;
    std::uint32_t readyNoiseImageIndex;
};

SwapchainObject::SwapchainObject(const GraphicsContext& context, SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain)
{
    // make swapchain
    swapchain = context.makeSwapchain(properties, oldSwapchain);
    swapchainImages = context.retrieveSwapchainImages(*swapchain);
    for (const auto& image : swapchainImages) {
        swapchainImageViews.push_back(context.makeImageView(image, properties.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
    }

    const std::size_t noiseFrameBuffersCount = 3;

    // hdr stage
    {
        // make multisampling buffer
        const vk::SampleCountFlagBits sampleCount = context.getMaxUsableSampleCount(2);
        const vk::Format hdrFormat = vk::Format::eR16G16B16A16Sfloat;
        multiSampleImage = context.makeMultiSampleImage(hdrFormat, properties.extent, 1, sampleCount);

        // make depth buffer
        depthImage = context.makeDepthImage(properties.extent, sampleCount);

        // make hdr float buffer
        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            hdrImages.push_back(context.makeImage(vk::SampleCountFlagBits::e1, 1, properties.extent, 1, hdrFormat,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor));
        }

        // make render pass
        std::vector<vk::SubpassDescription> subpasses;

        vk::AttachmentReference hdrAttachmentRef;
        hdrAttachmentRef.attachment = 0;
        hdrAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentReference depthAttachmentRef;
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference multiSampleAttachmentRef;
        multiSampleAttachmentRef.attachment = 2;
        multiSampleAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentReference bloomAttachmentRef;
        bloomAttachmentRef.attachment = 3;
        bloomAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDescription terrainSubpass;
        terrainSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        terrainSubpass.colorAttachmentCount = 1;
        terrainSubpass.pColorAttachments = &hdrAttachmentRef;
        terrainSubpass.pDepthStencilAttachment = &depthAttachmentRef;
        terrainSubpass.pResolveAttachments = &multiSampleAttachmentRef;
        subpasses.push_back(terrainSubpass);

        vk::SubpassDescription atmosphereSubpass;
        atmosphereSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        atmosphereSubpass.colorAttachmentCount = 1;
        atmosphereSubpass.pColorAttachments = &hdrAttachmentRef;
        atmosphereSubpass.pDepthStencilAttachment = &depthAttachmentRef;
        atmosphereSubpass.pResolveAttachments = &multiSampleAttachmentRef;
        subpasses.push_back(atmosphereSubpass);

        vk::SubpassDescription numbersSubpass;
        numbersSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        numbersSubpass.colorAttachmentCount = 1;
        numbersSubpass.pColorAttachments = &hdrAttachmentRef;
        numbersSubpass.pDepthStencilAttachment = nullptr;
        numbersSubpass.pResolveAttachments = &multiSampleAttachmentRef;
        subpasses.push_back(numbersSubpass);

        vk::SubpassDescription bloomSubpass;
        bloomSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        bloomSubpass.colorAttachmentCount = 1;
        bloomSubpass.pColorAttachments = &bloomAttachmentRef;
        bloomSubpass.pDepthStencilAttachment = nullptr;
        bloomSubpass.pResolveAttachments = nullptr;
        subpasses.push_back(bloomSubpass);

        // define dependencies
        std::vector<vk::SubpassDependency> dependencies;

        // terrain
        vk::SubpassDependency terrainDependency;
        terrainDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        terrainDependency.dstSubpass = 0;
        terrainDependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        terrainDependency.srcAccessMask = {};
        terrainDependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        terrainDependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies.push_back(terrainDependency);

        // atmosphere
        vk::SubpassDependency atmosphereDependency;
        atmosphereDependency.srcSubpass = 0;
        atmosphereDependency.dstSubpass = 1;
        atmosphereDependency.srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
        atmosphereDependency.srcAccessMask = {};
        atmosphereDependency.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        atmosphereDependency.dstAccessMask = {};
        dependencies.push_back(atmosphereDependency);

        // numbers
        vk::SubpassDependency numbersDependency;
        numbersDependency.srcSubpass = 1;
        numbersDependency.dstSubpass = 2;
        numbersDependency.srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
        numbersDependency.srcAccessMask = {};
        numbersDependency.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        numbersDependency.dstAccessMask = {};
        dependencies.push_back(numbersDependency);

        // bloom
        vk::SubpassDependency bloomDependency;
        bloomDependency.srcSubpass = 2;
        bloomDependency.dstSubpass = 3;
        bloomDependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        bloomDependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        bloomDependency.dstStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
        bloomDependency.dstAccessMask = {};
        dependencies.push_back(bloomDependency);

        // define attachments
        std::vector<vk::AttachmentDescription> attachments;

        // hdr color attachment
        vk::AttachmentDescription hdrAttachment;
        hdrAttachment.format = hdrFormat;
        hdrAttachment.samples = sampleCount;
        hdrAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        hdrAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        hdrAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        hdrAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        hdrAttachment.initialLayout = vk::ImageLayout::eUndefined;
        hdrAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachments.push_back(hdrAttachment);

        // depth attachment
        vk::AttachmentDescription depthAttachment;
        depthAttachment.format = depthImage.format;
        depthAttachment.samples = sampleCount;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
        depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        attachments.push_back(depthAttachment);

        // multisample resolve attachment
        vk::AttachmentDescription multiSampleAttachment;
        multiSampleAttachment.format = multiSampleImage.format;
        multiSampleAttachment.samples = vk::SampleCountFlagBits::e1;
        multiSampleAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
        multiSampleAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        multiSampleAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        multiSampleAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        multiSampleAttachment.initialLayout = vk::ImageLayout::eUndefined;
        multiSampleAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        attachments.push_back(multiSampleAttachment);

        // bloom color attachment
        vk::AttachmentDescription bloomAttachment;
        bloomAttachment.format = properties.surfaceFormat.format;
        bloomAttachment.samples = vk::SampleCountFlagBits::e1;
        bloomAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        bloomAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        bloomAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        bloomAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        bloomAttachment.initialLayout = vk::ImageLayout::eUndefined;
        bloomAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
        attachments.push_back(bloomAttachment);

        hdrRenderPass = context.makeRenderPass(subpasses, dependencies, attachments);

        // make descriptor sets
        descriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eCombinedImageSampler,
                vk::DescriptorType::eUniformBuffer },
            { vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
                vk::ShaderStageFlagBits::eAll },
            { 1, 1, noiseFrameBuffersCount, 3 });

        // make pipelines
        pipelineLayout = context.makePipelineLayout(*descriptorSet.layout);
        terrainPipeline = context.makePipeline(*pipelineLayout, properties.extent, *hdrRenderPass, 0, sampleCount,
            "shaders/planet.vert.spv", "shaders/planet.frag.spv", "shaders/planet.tesc.spv", "shaders/planet.tese.spv", nullptr,
            vk::PrimitiveTopology::ePatchList, true, false, {}, {});

        atmospherePipeline = context.makePipeline(*pipelineLayout, properties.extent, *hdrRenderPass, 1, sampleCount,
            "shaders/air.vert.spv", "shaders/air.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, true, false, {}, {});

        numbersPipeline = context.makePipeline(*pipelineLayout, properties.extent, *hdrRenderPass, 2, sampleCount,
            "shaders/numbers.vert.spv", "shaders/numbers.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, true, false, {}, {});

        bloomDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eCombinedImageSampler },
            { vk::ShaderStageFlagBits::eFragment },
            { 1 });

        bloomPipelineLayout = context.makePipelineLayout(*bloomDescriptorSet.layout);
        bloomPipeline = context.makePipeline(*bloomPipelineLayout, properties.extent, *hdrRenderPass, 3, vk::SampleCountFlagBits::e1,
            "shaders/bloom.vert.spv", "shaders/bloom.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, false, false, {}, {});

        // make hdr framebuffers
        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            hdrFramebuffers.push_back(context.makeFramebuffer(
                { *multiSampleImage.view, *depthImage.view, *hdrImages[i].view, *swapchainImageViews[i] },
                *hdrRenderPass, properties.extent));
        }

        // make present command buffers
        commandBuffers = context.allocateCommandBuffers(properties.imageCount);
    }

    // terrain construct stage
    {
        // make offscreen render target
        const vk::Extent2D noiseImageExtent = { properties.extent.width, properties.extent.height };
        const vk::Format noiseImageFormat = vk::Format::eR16G16B16A16Sfloat;
        const std::uint32_t noiseLayersCount = 2;

        for (std::size_t i = 0; i < noiseFrameBuffersCount; ++i) {
            ImageObject image = context.makeImage(vk::SampleCountFlagBits::e1, 1, noiseImageExtent, noiseLayersCount, noiseImageFormat,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
                vk::ImageAspectFlagBits::eColor);

            transitionImageLayout(*context.beginSingleTimeCommands(), *image.image, image.layerCount,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal, image.mipLevels);

            noiseImages.push_back(std::move(image));
        }

        // Host accessible scratch buffer
        terrain = context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eTransferDst,
            static_cast<vk::DeviceSize>(noiseImageExtent.width * noiseImageExtent.height * 8 * noiseLayersCount));

        noiseDescriptorSet = context.makeDescriptorSet(noiseFrameBuffersCount,
            { vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eStorageImage },
            { vk::ShaderStageFlagBits::eCompute, vk::ShaderStageFlagBits::eCompute },
            { 1, 1 });

        noisePipelineLayout = context.makePipelineLayout(*noiseDescriptorSet.layout);
        noisePipeline = context.makeComputePipeline(*noisePipelineLayout, "shaders/noise.comp.spv");

        noiseCommandBuffers = context.allocateCommandBuffers(noiseFrameBuffersCount);
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
    , m_inFlightFences(m_context.makeFences(maxFramesInFlight, true))
    , m_offscreenFences(m_context.makeFences(static_cast<std::uint32_t>(m_swapchain.noiseImages.size()), false))

    // make textures
    , m_textureImage(m_context.makeTextureImage("textures/texture.jpg"))

    // make sampler
    , m_sampler(m_context.makeTextureSampler(false))

    , m_parallelCount(200)
    , m_meridianCount(200)
    , m_mapBounds(m_swapchain.noiseImages.size())
{
    // make uniform buffers
    for (std::uint32_t i = 0; i < m_swapchainProps.imageCount; ++i) {
        m_uniformBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UniformBufferObject)));
        m_renderMapBoundsUniformBuffers.push_back(m_context.makeHostVisibleBuffer(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MapBoundsObject)));
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

static const std::uint64_t timeOut = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

void VulkanApplication::refreshSwapchain()
{
    m_context.device().waitIdle();

    m_updateOverallmap = true;
    m_updateHeightmap = true;

    for (const auto& fence : m_offscreenFences) {
        m_context.device().resetFences({ *fence });
    }

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
    std::array<vk::ClearValue, 4> clearValues{};

    // color buffer clear value
    clearValues[0].color.float32[0] = 0.0f;
    clearValues[0].color.float32[1] = 0.0f;
    clearValues[0].color.float32[2] = 0.0f;
    clearValues[0].color.float32[3] = 1.0f;

    // depth buffer clear value
    clearValues[1].depthStencil.depth = 1.0f;
    clearValues[1].depthStencil.stencil = 0;

    // color buffer clear value
    clearValues[2].color.float32[0] = 0.0f;
    clearValues[2].color.float32[1] = 0.0f;
    clearValues[2].color.float32[2] = 0.0f;
    clearValues[2].color.float32[3] = 1.0f;

    // color buffer clear value
    clearValues[3].color.float32[0] = 0.0f;
    clearValues[3].color.float32[1] = 0.0f;
    clearValues[3].color.float32[2] = 0.0f;
    clearValues[3].color.float32[3] = 1.0f;

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
            std::array<vk::WriteDescriptorSet, 5> descriptorWrites{};

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

            vk::DescriptorBufferInfo mapBoundsUniformBufferInfo{};
            mapBoundsUniformBufferInfo.buffer = *m_renderMapBoundsUniformBuffers[index].buffer;
            mapBoundsUniformBufferInfo.offset = 0;
            mapBoundsUniformBufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites[1].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &mapBoundsUniformBufferInfo;

            std::vector<vk::DescriptorImageInfo> noiseImageInfos{};
            for (auto const& image : m_swapchain.noiseImages) {
                vk::DescriptorImageInfo imageInfo;
                imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                imageInfo.imageView = *image.view;
                imageInfo.sampler = *m_sampler;
                noiseImageInfos.push_back(imageInfo);
            }

            descriptorWrites[2].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[2].descriptorCount = static_cast<std::uint32_t>(noiseImageInfos.size());
            descriptorWrites[2].pImageInfo = noiseImageInfos.data();

            std::vector<vk::DescriptorBufferInfo> numBufInfos{};
            for (auto const& buf : m_numberBuffers) {
                vk::DescriptorBufferInfo bufInfo;
                bufInfo.buffer = *buf.buffer;
                bufInfo.offset = 0;
                bufInfo.range = VK_WHOLE_SIZE;
                numBufInfos.push_back(bufInfo);
            }
            descriptorWrites[3].dstSet = m_swapchain.descriptorSet.sets[index];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[3].descriptorCount = static_cast<std::uint32_t>(numBufInfos.size());
            descriptorWrites[3].pBufferInfo = numBufInfos.data();

            vk::DescriptorImageInfo hdrImageInfo{};
            hdrImageInfo.imageLayout = vk::ImageLayout::eUndefined;
            hdrImageInfo.imageView = *m_swapchain.hdrImages[index].view;
            hdrImageInfo.sampler = *m_sampler;

            descriptorWrites[4].dstSet = m_swapchain.bloomDescriptorSet.sets[index];
            descriptorWrites[4].dstBinding = 0;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pImageInfo = &hdrImageInfo;

            m_context.device().updateDescriptorSets(descriptorWrites, {});

            // begin recording
            vk::CommandBufferBeginInfo beginInfo{};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
            beginInfo.pInheritanceInfo = nullptr;

            vk::CommandBuffer commandBuffer = *m_swapchain.commandBuffers[index];
            commandBuffer.begin(beginInfo);

            vk::RenderPassBeginInfo mainRenderPassInfo{};
            mainRenderPassInfo.renderPass = *m_swapchain.hdrRenderPass;
            mainRenderPassInfo.framebuffer = *m_swapchain.hdrFramebuffers[index];
            mainRenderPassInfo.renderArea.offset.x = 0;
            mainRenderPassInfo.renderArea.offset.y = 0;
            mainRenderPassInfo.renderArea.extent = m_swapchainProps.extent;
            mainRenderPassInfo.clearValueCount = clearValues.size();
            mainRenderPassInfo.pClearValues = clearValues.data();

            commandBuffer.beginRenderPass(mainRenderPassInfo, vk::SubpassContents::eInline);
            {
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.pipelineLayout, 0,
                    { m_swapchain.descriptorSet.sets[index] }, {});

                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.terrainPipeline);
                auto vertexCount = static_cast<std::uint32_t>(m_parallelCount * m_meridianCount * 4);
                commandBuffer.draw(vertexCount, 1, 0, 0);

                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.atmospherePipeline);
                commandBuffer.draw(4, 1, 0, 0);

                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.numbersPipeline);
                commandBuffer.draw(4, 1, 0, 0);

                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_swapchain.bloomPipelineLayout, 0,
                    { m_swapchain.bloomDescriptorSet.sets[index] }, {});
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_swapchain.bloomPipeline);
                commandBuffer.draw(4, 1, 0, 0);
            }
            commandBuffer.endRenderPass();

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
            static_cast<float>(m_swapchainProps.extent.width) / m_swapchainProps.extent.height, near, near * 100.0f);
        ubo.proj[1][1] *= -1; // invert Y axis

        ubo.iMVP = glm::inverse(ubo.proj * ubo.view * ubo.model);

        ubo.parallelCount = m_parallelCount;
        ubo.meridianCount = m_meridianCount;

        ubo.lightDir = glm::vec4(glm::normalize(glm::vec3(1, -1, 0)), 0);

        std::size_t nextOffscreenIndex = m_lastRenderedIndex + 1;
        if (nextOffscreenIndex >= m_swapchain.noiseImages.size()) {
            nextOffscreenIndex = 1;
        }
        if (m_renderingHeightmap && m_context.device().getFenceStatus(*m_offscreenFences[nextOffscreenIndex]) == vk::Result::eSuccess) {
            m_context.device().resetFences({ *m_offscreenFences[nextOffscreenIndex] });
            m_lastRenderedIndex = nextOffscreenIndex;
            m_renderingHeightmap = false;

            std::cout << "finished. current index: " << m_lastRenderedIndex << std::endl;
        }
        ubo.readyNoiseImageIndex = static_cast<std::uint32_t>(m_lastRenderedIndex);

        m_context.updateMemory(*m_uniformBuffers[imageIndex].memory, &ubo, sizeof(UniformBufferObject));
        m_context.updateMemory(*m_renderMapBoundsUniformBuffers[imageIndex].memory, &m_mapBounds[m_lastRenderedIndex], sizeof(MapBoundsObject));

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

                m_context.graphicsQueue2().submit({ submitInfo }, *m_offscreenFences[nextOffscreenIndex]);

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

        m_context.graphicsQueue2().submit({ submitInfo }, *m_offscreenFences[0]);

        m_updateOverallmap = false;
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
            m_currentFps = static_cast<std::uint32_t>(std::round(fps));

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

    //m_planetRotateAngle += dt / 4.0f * glm::radians(90.0f);

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
