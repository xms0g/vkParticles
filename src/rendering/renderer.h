#pragma once
#include <cstdint>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

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

	void createSwapchain();

	void createSwapchainImageViews();

	void recreateSwapchain();

	void createGraphicsPipeline();

	void createComputePipeline();

	void createCommandPool();

	void createUniformBuffers();

	void createShaderStorageBuffers();

	void createDescriptorPool();

	void createComputeDescriptorSets();

	void createComputeDescriptorSetLayout();

	void createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer,
		vk::raii::DeviceMemory& bufferMemory) const;

	void createCommandBuffers();

	void createComputeCommandBuffers();

	void recordGraphicsCommandBuffer(uint32_t imageIndex) const;

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

	bool checkDeviceSuitable(const vk::raii::PhysicalDevice& phyDevice);

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

	uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities);

	[[nodiscard]]
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

	void copyBuffer(const vk::raii::Buffer& srcBuffer, const vk::raii::Buffer& dstBuffer, vk::DeviceSize size) const;

	void updateUniformBuffer(uint32_t currentImage, float deltaTime) const;

	[[nodiscard]]
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

	static void transitionImageLayout(
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask,
		const vk::raii::CommandBuffer& commandBuffer) ;

	[[nodiscard]]
	vk::raii::CommandBuffer beginSingleTimeCommands() const;

	void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const;

	Window* mWindow{nullptr};

	vk::raii::Context mContext;
	vk::raii::Instance mInstance{nullptr};
	vk::raii::PhysicalDevice mPhysicalDevice{nullptr};
	vk::raii::Device mDevice{nullptr};
	vk::raii::Queue mGraphicsComputeQueue{nullptr};
	uint32_t mGraphicsAndComputeIndex{static_cast<uint32_t>(~0)};
	vk::raii::SurfaceKHR mSurface{nullptr};
	vk::raii::SwapchainKHR mSwapChain{nullptr};
	vk::SurfaceFormatKHR mSwapChainSurfaceFormat;
	vk::Extent2D mSwapChainExtent;
	std::vector<vk::Image> mSwapChainImages;
	std::vector<vk::raii::ImageView> mSwapChainImageViews;
	vk::raii::DescriptorSetLayout mComputeDescriptorSetLayout{nullptr};
	vk::raii::DescriptorPool mDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> mComputeDescriptorSets;
	vk::raii::PipelineLayout mGraphicsPipelineLayout{nullptr};
	vk::raii::PipelineLayout mComputePipelineLayout{nullptr};
	vk::raii::Pipeline mGraphicsPipeline{nullptr};
	vk::raii::Pipeline mComputePipeline{nullptr};
	std::vector<vk::raii::Buffer> mUniformBuffers;
	std::vector<vk::raii::DeviceMemory> mUniformBuffersMemory;
	std::vector<vk::raii::Buffer> mShaderStorageBuffers;
	std::vector<vk::raii::DeviceMemory> mShaderStorageBuffersMemory;
	std::vector<void*> mUniformBuffersMapped;
	vk::raii::CommandPool mCommandPool{nullptr};
	std::vector<vk::raii::CommandBuffer> mGraphicsCommandBuffers;
	std::vector<vk::raii::CommandBuffer> mComputeCommandBuffers;
	vk::raii::Semaphore mSemaphore{nullptr};
	uint64_t mTimelineValue{0};
	std::vector<vk::raii::Fence> mFences;
	uint32_t mFrameIndex{0};
	vk::raii::DebugUtilsMessengerEXT mDebugMessenger{nullptr};
};
