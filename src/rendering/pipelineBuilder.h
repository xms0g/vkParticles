#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include "vertex.hpp"
#include "shader.h"

class DescriptorSetLayout;
class Swapchain;
class Shader;

class PipelineBuilder {
public:
	explicit PipelineBuilder(const vk::raii::Device& device, const std::string& shaderPath);

	void reset();

	PipelineBuilder& addVertexShader(const std::string& entry);

	PipelineBuilder& addFragmentShader(const std::string& entry);

	PipelineBuilder& addComputeShader( const std::string& entry);

	PipelineBuilder& vertexInput(const VertexLayout& layout);

	PipelineBuilder& topology(vk::PrimitiveTopology topology);

	PipelineBuilder& viewportState(uint32_t viewportCount, uint32_t scissorCount);

	template<vk::DynamicState T>
	PipelineBuilder& dynamicState();

	PipelineBuilder& rasterizer();

	PipelineBuilder& multisampling();

	PipelineBuilder& alphaBlending();

	vk::raii::PipelineLayout createPipelineLayout(const vk::DescriptorSetLayout* dsc = nullptr , uint32_t count = 0) const;

	vk::raii::Pipeline buildGraphics(vk::SurfaceFormatKHR& surfaceFormat, const vk::raii::PipelineLayout& layout);

	vk::raii::Pipeline buildCompute(const vk::raii::PipelineLayout& layout) const;

private:
	Shader mShader;
	vk::PipelineInputAssemblyStateCreateInfo mInputAssembly;
	vk::PipelineViewportStateCreateInfo mViewportState;
	vk::PipelineRasterizationStateCreateInfo mRasterizer;
	vk::PipelineMultisampleStateCreateInfo mMultisampling;
	vk::PipelineColorBlendAttachmentState mColorBlendAttachment;
	vk::PipelineColorBlendStateCreateInfo mColorBlending;
	vk::PipelineDynamicStateCreateInfo mDynamicState;
	vk::PipelineVertexInputStateCreateInfo mVertexInputInfo;
	VertexLayout mVertexLayout;
	std::vector<vk::PipelineShaderStageCreateInfo> mShaderStages;
	std::vector<vk::DynamicState> mDynamicStates;
	const vk::raii::Device& mDevice;
};

template<vk::DynamicState T>
PipelineBuilder& PipelineBuilder::dynamicState() {
	mDynamicStates.push_back(T);
	return *this;
}

class Pipeline {
public:
	Pipeline() = default;

	vk::raii::PipelineLayout& layout() { return mPipelineLayout; }

	vk::raii::Pipeline& operator*() noexcept { return mPipeline; }
	const vk::raii::Pipeline& operator*() const noexcept { return mPipeline; }

protected:
	vk::raii::Pipeline mPipeline{nullptr};
	vk::raii::PipelineLayout mPipelineLayout{nullptr};
};

class GraphicsPipeline : public Pipeline {
public:
	GraphicsPipeline(PipelineBuilder& builder, Swapchain& swapchain, const VertexLayout& layout);
};

class ComputePipeline : public Pipeline {
public:
	ComputePipeline(PipelineBuilder& builder, DescriptorSetLayout& dscSetLayout, uint32_t dscSetLayoutCount);
};
