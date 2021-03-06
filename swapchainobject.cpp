#include "swapchainobject.h"

ou::SwapchainObject::SwapchainObject(const GraphicsContext& context, SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain)
{
    // make swapchain
    swapchain = context.makeSwapchain(properties, oldSwapchain);
    swapchainImages = context.retrieveSwapchainImages(*swapchain);
    for (const auto& image : swapchainImages) {
        swapchainImageViews.push_back(context.makeImageView(image, properties.surfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1));
    }

    const std::size_t noiseFrameBuffersCount = 3;
    const vk::Format hdrFormat = vk::Format::eR16G16B16A16Sfloat;

    // shadow rendering
    {
        vk::Extent2D shadowExtent = properties.extent;
        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            shadowImages.push_back(context.makeImage(vk::SampleCountFlagBits::e1, 1, shadowExtent, 1, vk::Format::eD32Sfloat,
                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eDepth));
        }

        vk::AttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 0;
        depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        std::vector<vk::SubpassDescription> subpasses;

        vk::SubpassDescription terrainSubpass{};
        terrainSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        terrainSubpass.colorAttachmentCount = 0;
        terrainSubpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpasses.push_back(terrainSubpass);

        std::vector<vk::SubpassDependency> dependencies;

        std::vector<vk::AttachmentDescription> attachments;

        vk::AttachmentDescription depthAttachment;
        depthAttachment.format = vk::Format::eD32Sfloat;
        depthAttachment.samples = vk::SampleCountFlagBits::e1;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
        depthAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        attachments.push_back(depthAttachment);

        shadowRenderPass = context.makeRenderPass(subpasses, dependencies, attachments);

        shadowPlanetDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eCombinedImageSampler },
            { vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment },
            { 1, 1, noiseFrameBuffersCount });

        shadowPlanetPipelineLayout = context.makePipelineLayout({ *shadowPlanetDescriptorSet.layout });
        shadowPlanetPipeline = context.makePipeline(*shadowPlanetPipelineLayout, shadowExtent, *shadowRenderPass, 0, vk::SampleCountFlagBits::e1,
            "shaders/planet.vert.spv", nullptr, "shaders/planet.tesc.spv", "shaders/planet.tese.spv", nullptr,
            vk::PrimitiveTopology::ePatchList, vk::CullModeFlagBits::eBack, true, BlendMode::Regular, false, {}, {});

        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            shadowFramebuffers.push_back(context.makeFramebuffer({ *shadowImages[i].view }, *shadowRenderPass, shadowExtent));
        }
    }

    // hdr stage
    {
        // make depth buffer
        depthImage = context.makeDepthImage(properties.extent, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eInputAttachment);

        // make hdr float buffer
        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            hdrImages.push_back(context.makeImage(vk::SampleCountFlagBits::e1, 1, properties.extent, 1, hdrFormat,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
                vk::ImageAspectFlagBits::eColor));
        }

        // make render pass
        vk::AttachmentReference hdrAttachmentRef{};
        hdrAttachmentRef.attachment = 0;
        hdrAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference inputAttachmentRef{};
        inputAttachmentRef.attachment = 1;
        inputAttachmentRef.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

        std::vector<vk::SubpassDescription> subpasses;

        vk::SubpassDescription terrainSubpass{};
        terrainSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        terrainSubpass.colorAttachmentCount = 1;
        terrainSubpass.pColorAttachments = &hdrAttachmentRef;
        terrainSubpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpasses.push_back(terrainSubpass);

        vk::SubpassDescription atmosphereSubpass{};
        atmosphereSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        atmosphereSubpass.colorAttachmentCount = 1;
        atmosphereSubpass.pColorAttachments = &hdrAttachmentRef;
        atmosphereSubpass.inputAttachmentCount = 1;
        atmosphereSubpass.pInputAttachments = &inputAttachmentRef;
        subpasses.push_back(atmosphereSubpass);

        // define dependencies
        std::vector<vk::SubpassDependency> dependencies;

        // terrain
        vk::SubpassDependency terrainDependency{};
        terrainDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        terrainDependency.dstSubpass = 0;
        terrainDependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        terrainDependency.srcAccessMask = {};
        terrainDependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        terrainDependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        terrainDependency.dependencyFlags = vk::DependencyFlagBits::eByRegion;
        dependencies.push_back(terrainDependency);

        // atmosphere
        vk::SubpassDependency atmosphereDependency{};
        atmosphereDependency.srcSubpass = 0;
        atmosphereDependency.dstSubpass = 1;
        atmosphereDependency.srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
        atmosphereDependency.srcAccessMask = {};
        atmosphereDependency.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        atmosphereDependency.dstAccessMask = {};
        atmosphereDependency.dependencyFlags = vk::DependencyFlagBits::eByRegion;
        dependencies.push_back(atmosphereDependency);

        // define attachments
        std::vector<vk::AttachmentDescription> attachments;

        // hdr color attachment
        vk::AttachmentDescription hdrAttachment{};
        hdrAttachment.format = hdrFormat;
        hdrAttachment.samples = vk::SampleCountFlagBits::e1;
        hdrAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        hdrAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        hdrAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        hdrAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        hdrAttachment.initialLayout = vk::ImageLayout::eUndefined;
        hdrAttachment.finalLayout = vk::ImageLayout::eGeneral;
        attachments.push_back(hdrAttachment);

        // depth attachment
        vk::AttachmentDescription depthAttachment{};
        depthAttachment.format = depthImage.format;
        depthAttachment.samples = vk::SampleCountFlagBits::e1;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
        depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        attachments.push_back(depthAttachment);

        hdrRenderPass = context.makeRenderPass(subpasses, dependencies, attachments);

        planetDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eUniformBuffer,
                vk::DescriptorType::eCombinedImageSampler },
            { vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eAll,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment },
            { 1, 1, noiseFrameBuffersCount });

        shadowMapDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eCombinedImageSampler }, { vk::ShaderStageFlagBits::eFragment }, { 1 });

        deferredDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eInputAttachment }, { vk::ShaderStageFlagBits::eFragment }, { 1 });

        // make pipelines
        planetPipelineLayout = context.makePipelineLayout({ *planetDescriptorSet.layout, *shadowMapDescriptorSet.layout });
        planetPipeline = context.makePipeline(*planetPipelineLayout, properties.extent, *hdrRenderPass, 0, vk::SampleCountFlagBits::e1,
            "shaders/planet.vert.spv", "shaders/planet.frag.spv", "shaders/planet.tesc.spv", "shaders/planet.tese.spv", nullptr,
            vk::PrimitiveTopology::ePatchList, vk::CullModeFlagBits::eBack, true, BlendMode::Regular, false, {}, {});

        atmospherePipelineLayout = context.makePipelineLayout(
            { *planetDescriptorSet.layout, *shadowMapDescriptorSet.layout, *deferredDescriptorSet.layout });
        atmospherePipeline = context.makePipeline(*atmospherePipelineLayout, properties.extent, *hdrRenderPass, 1, vk::SampleCountFlagBits::e1,
            "shaders/air.vert.spv", "shaders/air.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, vk::CullModeFlagBits::eFront, false, BlendMode::Additive, false, {}, {});

        // make hdr framebuffers
        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            framebuffers.push_back(context.makeFramebuffer(
                { *hdrImages[i].view, *depthImage.view },
                *hdrRenderPass, properties.extent));
        }
    }

    // bloom stage
    {
        const vk::Extent2D bloomExtent{ properties.extent.width, properties.extent.height / 2 };

        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            brightImages.push_back(context.makeImage(vk::SampleCountFlagBits::e1, 1, bloomExtent, 1, vk::Format::eB8G8R8A8Unorm,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
                vk::ImageAspectFlagBits::eColor));

            bloomImages.push_back(context.makeImage(vk::SampleCountFlagBits::e1, 1, bloomExtent, 1, vk::Format::eB8G8R8A8Unorm,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                vk::ImageAspectFlagBits::eColor));
        }

        downsampleDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eStorageImage, vk::DescriptorType::eStorageImage },
            { vk::ShaderStageFlagBits::eCompute, vk::ShaderStageFlagBits::eCompute },
            { 1, 1 });

        bloomDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eStorageImage, vk::DescriptorType::eStorageImage },
            { vk::ShaderStageFlagBits::eCompute, vk::ShaderStageFlagBits::eCompute },
            { 1, 1 });

        downsamplePipelineLayout = context.makePipelineLayout({ *downsampleDescriptorSet.layout });
        downsamplePipeline = context.makeComputePipeline(*downsamplePipelineLayout, "shaders/down.comp.spv");

        bloomPipelineLayout = context.makePipelineLayout({ *bloomDescriptorSet.layout });
        bloomHPipeline = context.makeComputePipeline(*bloomPipelineLayout, "shaders/bloomh.comp.spv");
        bloomVPipeline = context.makeComputePipeline(*bloomPipelineLayout, "shaders/bloomv.comp.spv");
    }

    // present stage
    {
        vk::AttachmentReference presentAttachmentRef;
        presentAttachmentRef.attachment = 0;
        presentAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        std::vector<vk::SubpassDescription> subpasses;

        vk::SubpassDescription sceneSubpass{};
        sceneSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        sceneSubpass.colorAttachmentCount = 1;
        sceneSubpass.pColorAttachments = &presentAttachmentRef;
        sceneSubpass.pDepthStencilAttachment = nullptr;
        sceneSubpass.pResolveAttachments = nullptr;
        subpasses.push_back(sceneSubpass);

        vk::SubpassDescription numbersSubpass{};
        numbersSubpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        numbersSubpass.colorAttachmentCount = 1;
        numbersSubpass.pColorAttachments = &presentAttachmentRef;
        numbersSubpass.pDepthStencilAttachment = nullptr;
        numbersSubpass.pResolveAttachments = nullptr;
        subpasses.push_back(numbersSubpass);

        std::vector<vk::SubpassDependency> dependencies;

        // scene
        vk::SubpassDependency sceneDependency{};
        sceneDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        sceneDependency.dstSubpass = 0;
        sceneDependency.srcStageMask = vk::PipelineStageFlagBits::eAllCommands;
        sceneDependency.srcAccessMask = {};
        sceneDependency.dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
        sceneDependency.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        dependencies.push_back(sceneDependency);

        // numbers
        vk::SubpassDependency numbersDependency{};
        numbersDependency.srcSubpass = 0;
        numbersDependency.dstSubpass = 1;
        numbersDependency.srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
        numbersDependency.srcAccessMask = {};
        numbersDependency.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        numbersDependency.dstAccessMask = {};
        dependencies.push_back(numbersDependency);

        std::vector<vk::AttachmentDescription> attachments;

        // present attachment
        vk::AttachmentDescription presentAttachment{};
        presentAttachment.format = properties.surfaceFormat.format;
        presentAttachment.samples = vk::SampleCountFlagBits::e1;
        presentAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
        presentAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        presentAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        presentAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        presentAttachment.initialLayout = vk::ImageLayout::eUndefined;
        presentAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
        attachments.push_back(presentAttachment);

        presentRenderPass = context.makeRenderPass(subpasses, dependencies, attachments);

        sceneDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eCombinedImageSampler, vk::DescriptorType::eCombinedImageSampler },
            { vk::ShaderStageFlagBits::eFragment, vk::ShaderStageFlagBits::eFragment },
            { 1, 1 });

        scenePipelineLayout = context.makePipelineLayout({ *sceneDescriptorSet.layout });
        scenePipeline = context.makePipeline(*scenePipelineLayout, properties.extent, *presentRenderPass, 0, vk::SampleCountFlagBits::e1,
            "shaders/scene.vert.spv", "shaders/scene.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, vk::CullModeFlagBits::eFront, false, BlendMode::None, false, {}, {});

        numbersDescriptorSet = context.makeDescriptorSet(properties.imageCount,
            { vk::DescriptorType::eUniformBuffer }, { vk::ShaderStageFlagBits::eFragment }, { 3 });

        numbersPipelineLayout = context.makePipelineLayout({ *numbersDescriptorSet.layout });
        numbersPipeline = context.makePipeline(*numbersPipelineLayout, properties.extent, *presentRenderPass, 1, vk::SampleCountFlagBits::e1,
            "shaders/numbers.vert.spv", "shaders/numbers.frag.spv", nullptr, nullptr, nullptr,
            vk::PrimitiveTopology::eTriangleFan, vk::CullModeFlagBits::eFront, false, BlendMode::Regular, false, {}, {});

        for (std::size_t i = 0; i < properties.imageCount; ++i) {
            presentFramebuffers.push_back(context.makeFramebuffer({ *swapchainImageViews[i] }, *presentRenderPass, properties.extent));
        }
    }

    // make present command buffers
    commandBuffers = context.allocateCommandBuffers(properties.imageCount);

    // terrain construct stage
    {
        // make offscreen render target
        const vk::Extent2D noiseImageExtent = { properties.extent.width, properties.extent.height };
        const vk::Format noiseImageFormat = vk::Format::eR32G32B32A32Sfloat;
        const std::uint32_t noiseLayersCount = 2;

        noiseFences = context.makeFences(noiseFrameBuffersCount, false);

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
            static_cast<vk::DeviceSize>(noiseImageExtent.width * noiseImageExtent.height * 16 * noiseLayersCount));

        noiseDescriptorSet = context.makeDescriptorSet(noiseFrameBuffersCount,
            { vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eStorageImage },
            { vk::ShaderStageFlagBits::eCompute, vk::ShaderStageFlagBits::eCompute },
            { 1, 1 });

        noisePipelineLayout = context.makePipelineLayout({ *noiseDescriptorSet.layout });
        noisePipeline = context.makeComputePipeline(*noisePipelineLayout, "shaders/noise.comp.spv");

        noiseCommandBuffers = context.allocateCommandBuffers(noiseFrameBuffersCount);
    }
}
