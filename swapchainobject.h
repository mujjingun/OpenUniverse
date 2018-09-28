#ifndef SWAPCHAINOBJECT_H
#define SWAPCHAINOBJECT_H

#include <vector>

#include "graphicscontext.h"

namespace ou {

const std::uint32_t maxSampleCount = 2;

// set of elements that need to be recreated when the window gets resized
struct SwapchainObject {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages{};
    std::vector<vk::UniqueImageView> swapchainImageViews{};

    // shadow mapping stage
    std::vector<ImageObject> shadowImages{};

    vk::UniqueRenderPass shadowRenderPass;

    DescriptorSetObject shadowPlanetDescriptorSet;

    vk::UniquePipelineLayout shadowPlanetPipelineLayout;
    vk::UniquePipeline shadowPlanetPipeline;

    std::vector<vk::UniqueFramebuffer> shadowFramebuffers{};

    // main render stage
    ImageObject multiSampleImage;
    ImageObject depthImage;
    std::vector<ImageObject> hdrImages{};

    vk::UniqueRenderPass hdrRenderPass;

    DescriptorSetObject planetDescriptorSet;
    DescriptorSetObject shadowMapDescriptorSet;

    vk::UniquePipelineLayout planetPipelineLayout;
    vk::UniquePipeline planetPipeline;
    vk::UniquePipeline atmospherePipeline;

    std::vector<vk::UniqueFramebuffer> framebuffers{};

    // bloom shader
    std::vector<ImageObject> brightImages{};
    std::vector<ImageObject> bloomImages{};

    DescriptorSetObject downsampleDescriptorSet;
    DescriptorSetObject bloomDescriptorSet;

    vk::UniquePipelineLayout downsamplePipelineLayout;
    vk::UniquePipeline downsamplePipeline;

    vk::UniquePipelineLayout bloomPipelineLayout;
    vk::UniquePipeline bloomHPipeline, bloomVPipeline;

    // present shader
    vk::UniqueRenderPass presentRenderPass;

    DescriptorSetObject sceneDescriptorSet;
    DescriptorSetObject numbersDescriptorSet;

    vk::UniquePipelineLayout scenePipelineLayout;
    vk::UniquePipeline scenePipeline;

    vk::UniquePipelineLayout numbersPipelineLayout;
    vk::UniquePipeline numbersPipeline;

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
