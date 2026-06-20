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

	void createColorResources();

	void recreateSwapchain();

	void createGraphicsPipeline();

	void createComputePipeline();

	void createCommandPool();

	void createUniformBuffers();

	void createShaderStorageBuffers();

	void createDescriptorPool();

	void createComputeDescriptorSets();

	void createComputeDescriptorSetLayout();

	void createDepthResources();

	void createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer,
		vk::raii::DeviceMemory& bufferMemory) const;

	void createCommandBuffers();

	void createComputeCommandBuffers();

	void recordCommandBuffer(uint32_t imageIndex) const;

	void recordComputeCommandBuffer();

	void createSyncObjects();

	// Getters
	void getPhysicalDevice();

	// Support Functions
	std::vector<const char*> getRequiredInstanceExtensions();

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

	[[nodiscard]]
	vk::Format findDepthFormat() const;

	[[nodiscard]]
	vk::Format findSupportedFormat(
		const std::vector<vk::Format>& candidates,
		vk::ImageTiling tiling,
		vk::FormatFeatureFlags features) const;

	bool hasStencilComponent(vk::Format format);

	void copyBuffer(const vk::raii::Buffer& srcBuffer, const vk::raii::Buffer& dstBuffer, vk::DeviceSize size) const;

	void copyBufferToImage(const vk::raii::Buffer& buffer, const vk::raii::Image& image, uint32_t width,
	                       uint32_t height) const;

	void updateUniformBuffer(uint32_t currentImage, float deltaTime) const;

	[[nodiscard]]
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

	void createImage(
		uint32_t width,
		uint32_t height,
		uint32_t mipLevels,
		vk::SampleCountFlagBits numSamples,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Image& image,
		vk::raii::DeviceMemory& imageMemory) const;

	[[nodiscard]]
	vk::raii::ImageView createImageView(
		const vk::raii::Image& image,
		vk::Format format,
		vk::ImageAspectFlags aspectFlags,
		uint32_t mipLevels) const;

	void createTextureSampler();

	void generateMipmaps(
		const vk::raii::Image &image,
		vk::Format imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels) const;

	void transitionImageLayout(
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask,
		const vk::raii::CommandBuffer& commandBuffer) const;

	[[nodiscard]]
	vk::SampleCountFlagBits getMaxUsableSampleCount() const;

	[[nodiscard]]
	vk::raii::CommandBuffer beginSingleTimeCommands() const;

	void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const;

	Window* mWindow{nullptr};

	vk::raii::Context mContext;
	vk::raii::Instance mInstance{nullptr};
	vk::raii::PhysicalDevice mPhysicalDevice{nullptr};
	vk::raii::Device mDevice{nullptr};
	vk::raii::Queue mGraphicsQueue{nullptr};
	uint32_t mGraphicsQueueFamilyIndex{static_cast<uint32_t>(~0)};
	vk::raii::SurfaceKHR mSurface{nullptr};
	vk::raii::SwapchainKHR mSwapChain{nullptr};
	vk::SurfaceFormatKHR mSwapChainSurfaceFormat;
	vk::Extent2D mSwapChainExtent;
	std::vector<vk::Image> mSwapChainImages;
	std::vector<vk::raii::ImageView> mSwapChainImageViews;
	vk::raii::DescriptorSetLayout mComputeDescriptorSetLayout{nullptr};
	vk::raii::DescriptorPool mDescriptorPool{nullptr};
	std::vector<vk::raii::DescriptorSet> mComputeDescriptorSets;
	vk::raii::PipelineLayout mPipelineLayout{nullptr};
	vk::raii::PipelineLayout mComputePipelineLayout{nullptr};
	vk::raii::Pipeline mGraphicsPipeline{nullptr};
	vk::raii::Pipeline mComputePipeline{nullptr};
	vk::raii::Buffer mVertexBuffer{nullptr};
	vk::raii::DeviceMemory mVertexBufferMemory{nullptr};
	vk::raii::Buffer mIndexBuffer{nullptr};
	vk::raii::DeviceMemory mIndexBufferMemory{nullptr};
	std::vector<vk::raii::Buffer> mUniformBuffers;
	std::vector<vk::raii::DeviceMemory> mUniformBuffersMemory;
	std::vector<vk::raii::Buffer> mShaderStorageBuffers;
	std::vector<vk::raii::DeviceMemory> mShaderStorageBuffersMemory;
	std::vector<void*> mUniformBuffersMapped;
	vk::raii::CommandPool mCommandPool{nullptr};
	std::vector<vk::raii::CommandBuffer> mCommandBuffers;
	std::vector<vk::raii::CommandBuffer> mComputeCommandBuffers;
	vk::raii::Semaphore mSemaphore{nullptr};
	uint64_t mTimelineValue{0};
	std::vector<vk::raii::Fence> mFences;
	uint32_t mFrameIndex{0};
	vk::raii::Image mTextureImage{nullptr};
	vk::raii::DeviceMemory mTextureImageMemory{nullptr};
	vk::raii::ImageView mTextureImageView{nullptr};
	vk::raii::Sampler mTextureSampler{nullptr};
	vk::raii::Image mDepthImage{nullptr};
	vk::raii::DeviceMemory mDepthImageMemory{nullptr};
	vk::raii::ImageView mDepthImageView{nullptr};
	vk::SampleCountFlagBits mMSAASamples;
	vk::raii::Image mColorImage{nullptr};
	vk::raii::DeviceMemory mColorImageMemory{nullptr};
	vk::raii::ImageView mColorImageView{nullptr};
	vk::raii::DebugUtilsMessengerEXT mDebugMessenger{nullptr};
};
