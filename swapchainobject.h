#ifndef SWAPCHAINOBJECT_H
#define SWAPCHAINOBJECT_H

#include <vector>

#include "graphicscontext.h"

namespace ou {

// set of elements that need to be recreated when the window gets resized
struct SwapchainObject {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages{};
    std::vector<vk::UniqueImageView> swapchainImageViews{};

    // main render stage
    ImageObject multiSampleImage;
    ImageObject depthImage;
    std::vector<ImageObject> hdrImages{};
    std::vector<ImageObject> scaledHdrImages{};
    std::vector<ImageObject> bloomImages{};

    vk::UniqueRenderPass hdrRenderPass;

    DescriptorSetObject descriptorSet;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline terrainPipeline;
    vk::UniquePipeline atmospherePipeline;

    std::vector<vk::UniqueFramebuffer> framebuffers{};

    // present & bloom shader
    vk::UniqueRenderPass presentRenderPass;
    vk::UniqueRenderPass bloomRenderPass;

    DescriptorSetObject bloomHDescriptorSet;
    DescriptorSetObject bloomVDescriptorSet;

    DescriptorSetObject sceneDescriptorSet;
    DescriptorSetObject numbersDescriptorSet;

    vk::UniquePipelineLayout bloomHPipelineLayout;
    vk::UniquePipeline bloomHPipeline;

    vk::UniquePipelineLayout bloomVPipelineLayout;
    vk::UniquePipeline bloomVPipeline;

    vk::UniquePipelineLayout scenePipelineLayout;
    vk::UniquePipeline scenePipeline;

    vk::UniquePipelineLayout numbersPipelineLayout;
    vk::UniquePipeline numbersPipeline;

    std::vector<vk::UniqueFramebuffer> bloomFramebuffers{};
    std::vector<vk::UniqueFramebuffer> presentFramebuffers{};

    std::vector<vk::UniqueCommandBuffer> commandBuffers{};

    // noise render pass
    std::vector<vk::UniqueFence> noiseFences;

    std::vector<ImageObject> noiseImages{};
    BufferObject terrain;

    DescriptorSetObject noiseDescriptorSet;
    vk::UniquePipelineLayout noisePipelineLayout;
    vk::UniquePipeline noisePipeline;

    std::vector<vk::UniqueCommandBuffer> noiseCommandBuffers{};

    SwapchainObject() = default;
    SwapchainObject(GraphicsContext const& context, SwapchainProperties const& properties, vk::SwapchainKHR oldSwapchain = nullptr);
};

}

#endif // SWAPCHAINOBJECT_H
