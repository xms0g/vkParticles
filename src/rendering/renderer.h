#pragma once
#include <cstdint>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include "swapchain.h"

class Buffer;
class Window;

class Renderer {
public:
	Renderer();

	~Renderer();

	int init(Window* window);

	void render(float deltaTime);

	void waitIdle() const;

private:
	void createInstance();

	void setupDebugMessenger();

	void createSurface();

	void createLogicalDevice();

	void createGraphicsPipeline();

	void createComputePipeline();

	void createCommandPool();

	void createUniformBuffers();

	void createShaderStorageBuffers();

	void createDescriptorPool();

	void createComputeDescriptorSets();

	void createComputeDescriptorSetLayout();

	void createCommandBuffers();

	void recordGraphicsCommandBuffer(uint32_t imageIndex);

	void recordComputeCommandBuffer();

	void createSyncObjects();

	// Getters
	void getPhysicalDevice();

	// Support Functions
	static std::vector<const char*> getRequiredInstanceExtensions();

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
		vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData);

	static bool checkDeviceSuitable(const vk::raii::PhysicalDevice& phyDevice);

	void copyBuffer(const Buffer& dstBuffer, const Buffer& srcBuffer, vk::DeviceSize size) const;

	void updateUniformBuffer(uint32_t currentImage, float deltaTime) const;

	[[nodiscard]]
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

	[[nodiscard]]
	vk::raii::CommandBuffer beginSingleTimeCommands() const;

	void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const;

	struct UniformBufferObject {
		float deltaTime{1.0f};
	};

	Window* mWindow{nullptr};

	vk::raii::Context mContext;
	vk::raii::Instance mInstance{nullptr};
	vk::raii::PhysicalDevice mPhysicalDevice{nullptr};
	vk::raii::Device mDevice{nullptr};
	vk::raii::Queue mQueue{nullptr};
	uint32_t mQueueIndex{static_cast<uint32_t>(~0)};
	vk::raii::SurfaceKHR mSurface{nullptr};
	Swapchain mSwapChain;
	vk::raii::DescriptorSetLayout mComputeDescriptorSetLayout{nullptr};
	vk::raii::DescriptorPool mDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> mComputeDescriptorSets;
	vk::raii::PipelineLayout mGraphicsPipelineLayout{nullptr};
	vk::raii::PipelineLayout mComputePipelineLayout{nullptr};
	vk::raii::Pipeline mGraphicsPipeline{nullptr};
	vk::raii::Pipeline mComputePipeline{nullptr};
	std::vector<Buffer> mUniformBuffers;
	std::vector<Buffer> mShaderStorageBuffers;
	vk::raii::CommandPool mCommandPool{nullptr};
	std::vector<vk::raii::CommandBuffer> mGraphicsCommandBuffers;
	std::vector<vk::raii::CommandBuffer> mComputeCommandBuffers;
	vk::raii::Semaphore mSemaphore{nullptr};
	uint64_t mTimelineValue{0};
	uint32_t mFrameIndex{0};
	std::vector<vk::raii::Fence> mFences;
	vk::raii::DebugUtilsMessengerEXT mDebugMessenger{nullptr};
};
