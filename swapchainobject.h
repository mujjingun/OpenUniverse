#ifndef SWAPCHAINOBJECT_H
#define SWAPCHAINOBJECT_H

#include <vector>

#include "graphicscontext.h"

namespace ou {

const std::size_t bloomCount = 3;

// set of elements that need to be recreated when the window gets resized
struct SwapchainObject {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages{};
    std::vector<vk::UniqueImageView> swapchainImageViews{};

    // main render stage
    ImageObject multiSampleImage;
    ImageObject depthImage;
    std::vector<ImageObject> hdrImages{};
    std::vector<ImageObject> colorImages{};
    std::vector<std::vector<ImageObject>> scaledHdrImages{bloomCount};
    std::vector<std::vector<ImageObject>> bloomImages{bloomCount};

    vk::UniqueRenderPass hdrRenderPass;

    DescriptorSetObject descriptorSet;

    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline terrainPipeline;
    vk::UniquePipeline atmospherePipeline;

    std::vector<vk::UniqueFramebuffer> framebuffers{};
    std::vector<std::vector<vk::UniqueFramebuffer>> bloomFramebuffers{bloomCount};

    // present & bloom shader
    vk::UniqueRenderPass presentRenderPass;
    vk::UniqueRenderPass bloomRenderPass;

    std::vector<DescriptorSetObject> bloomHDescriptorSet{bloomCount};

    DescriptorSetObject sceneDescriptorSet;
    DescriptorSetObject numbersDescriptorSet;

    std::vector<vk::UniquePipelineLayout> bloomHPipelineLayout{bloomCount};
    std::vector<vk::UniquePipeline> bloomHPipeline{bloomCount};

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
