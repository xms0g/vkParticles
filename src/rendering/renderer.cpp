#include "renderer.h"
#include <cstring>
#include <set>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <random>
#include <unordered_set>
#include <GLFW/glfw3.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "particle.hpp"
#include "buffer.h"
#include "swapchain.h"
#include "deviceExtension.hpp"
#include "validation.hpp"
#include "../core/window.h"
#include "../config/config.hpp"
#include "../io/filesystem.h"

Renderer::Renderer() = default;

Renderer::~Renderer() = default;

int Renderer::init(Window* window) {
	mWindow = window;

	try {
		createInstance();
		setupDebugMessenger();
		createSurface();
		getPhysicalDevice();
		createLogicalDevice();
		mSwapChain.create(mSurface, mDevice, mPhysicalDevice, mWindow->nativeHandle());
		createComputeDescriptorSetLayout();
		createGraphicsPipeline();
		createComputePipeline();
		createCommandPool();
		createShaderStorageBuffers();
		createUniformBuffers();
		createDescriptorPool();
		createComputeDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	} catch (const std::runtime_error& e) {
		throw std::runtime_error(e.what());
	}
	return 0;
}

void Renderer::render(const float deltaTime) {
	uint32_t imageIndex = mSwapChain.acquireNextImage(mFences[mFrameIndex]);

	auto fenceResult = mDevice.waitForFences(*mFences[mFrameIndex], vk::True, UINT64_MAX);
	if (fenceResult != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to wait for fence!");
	}

	mDevice.resetFences(*mFences[mFrameIndex]);

	uint64_t computeWaitValue = mTimelineValue;
	uint64_t computeSignalValue = ++mTimelineValue;
	uint64_t graphicsWaitValue = computeSignalValue;
	uint64_t graphicsSignalValue = ++mTimelineValue;

	updateUniformBuffer(mFrameIndex, deltaTime);

	{
		recordComputeCommandBuffer();
		// Submit compute work
		vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
			.waitSemaphoreValueCount = 1,
			.pWaitSemaphoreValues = &computeWaitValue,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = &computeSignalValue
		};

		vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};

		const vk::SubmitInfo computeSubmitInfo{
			.pNext = &computeTimelineInfo,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*mSemaphore,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &*mComputeCommandBuffers[mFrameIndex],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*mSemaphore
		};

		mQueue.submit(computeSubmitInfo, nullptr);
	}
	{
		// Record graphics command buffer
		recordGraphicsCommandBuffer(imageIndex);

		// Submit graphics work (waits for compute to finish)
		vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;
		vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
			.waitSemaphoreValueCount = 1,
			.pWaitSemaphoreValues = &graphicsWaitValue,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = &graphicsSignalValue
		};

		const vk::SubmitInfo graphicsSubmitInfo{
			.pNext = &graphicsTimelineInfo,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*mSemaphore,
			.pWaitDstStageMask = &waitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &*mGraphicsCommandBuffers[mFrameIndex],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*mSemaphore
		};

		mQueue.submit(graphicsSubmitInfo, nullptr);

		// Present the image (wait for graphics to finish)
		const vk::SemaphoreWaitInfo waitInfo{
			.semaphoreCount = 1,
			.pSemaphores = &*mSemaphore,
			.pValues = &graphicsSignalValue
		};

		// Wait for graphics to complete before presenting
		auto result = mDevice.waitSemaphores(waitInfo, UINT64_MAX);
		if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to wait for semaphore!");
		}

		const vk::PresentInfoKHR presentInfo{
			.waitSemaphoreCount = 0, // No binary semaphores needed
			.pWaitSemaphores = nullptr,
			.swapchainCount = 1,
			.pSwapchains = &*mSwapChain.native(),
			.pImageIndices = &imageIndex
		};

		result = mQueue.presentKHR(presentInfo);
		// Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as a result
		// here and does not need to be caught by an exception.
		if ((result == vk::Result::eSuboptimalKHR) ||
		    (result == vk::Result::eErrorOutOfDateKHR) ||
		    mWindow->windowResized()) {
			mWindow->windowResized(false);
			mSwapChain.recreate(mSurface, mDevice, mPhysicalDevice, mWindow->nativeHandle());
		} else {
			// There are no other success codes than eSuccess; on any error code, presentKHR already threw an exception.
			assert(result == vk::Result::eSuccess);
		}
	}
	mFrameIndex = (mFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::waitIdle() const {
	mDevice.waitIdle();
}

void Renderer::createInstance() {
	constexpr vk::ApplicationInfo appInfo{
		.pApplicationName = "Particle Simulation",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "No Engine",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = vk::ApiVersion14
	};

	if (enableValidationLayers) {
		std::unordered_set<std::string> supportedValidationLayers;
		for (const auto& layer: mContext.enumerateInstanceLayerProperties()) {
			supportedValidationLayers.insert(layer.layerName);
		}

		for (auto& layer: validationLayers) {
			if (!supportedValidationLayers.contains(layer)) {
				throw std::runtime_error("Required Validation Layer not supported: " + std::string(layer));
			}
		}
	}

	const auto glfwExtensions = getRequiredInstanceExtensions();

	std::unordered_set<std::string> supportedExtensions;
	for (const auto& [extensionName, specVersion]: mContext.enumerateInstanceExtensionProperties()) {
		supportedExtensions.insert(extensionName);
	}

	std::vector<const char*> requiredExtensions;
	for (const auto& extension: glfwExtensions) {
		if (!supportedExtensions.contains(extension)) {
			throw std::runtime_error("Required GLFW extension not supported: " + std::string(extension));
		}
		requiredExtensions.emplace_back(extension);
	}

	vk::InstanceCreateFlagBits flags{0};
#ifdef __APPLE__
	requiredExtensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
	flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

	const vk::InstanceCreateInfo createInfo{
		.flags = flags,
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
		.ppEnabledLayerNames = validationLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data()
	};

	mInstance = vk::raii::Instance(mContext, createInfo);
}

void Renderer::setupDebugMessenger() {
	if constexpr (!enableValidationLayers) {
		return;
	}

	constexpr vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
	};

	constexpr vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
	};

	constexpr vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags,
		.messageType = messageTypeFlags,
		.pfnUserCallback = &debugCallback
	};

	mDebugMessenger = mInstance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void Renderer::createSurface() {
	VkSurfaceKHR surface;
	if (glfwCreateWindowSurface(*mInstance, mWindow->nativeHandle(), nullptr, &surface) != 0) {
		throw std::runtime_error("Failed to create window surface!");
	}

	mSurface = vk::raii::SurfaceKHR(mInstance, surface);
}

void Renderer::createLogicalDevice() {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = mPhysicalDevice.getQueueFamilyProperties();

	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics &&
		    queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute &&
		    mPhysicalDevice.getSurfaceSupportKHR(i, *mSurface)) {
			mQueueIndex = i;
			break;
		}
	}

	if (mQueueIndex == ~0) {
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
	}

	// Create a chain of feature structures
	vk::StructureChain<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> featureChain = {
		{.features = {.samplerAnisotropy = true}},
		{.synchronization2 = true, .dynamicRendering = true},
		{.extendedDynamicState = true},
		{.timelineSemaphore = true}
	};

	float queuePriority = 0.5f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
		.queueFamilyIndex = mQueueIndex,
		.queueCount = 1,
		.pQueuePriorities = &queuePriority
	};

	vk::DeviceCreateInfo deviceCreateInfo{
		.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
		.ppEnabledExtensionNames = deviceExtensions.data()
	};

	mDevice = vk::raii::Device(mPhysicalDevice, deviceCreateInfo);
	mQueue = vk::raii::Queue(mDevice, mQueueIndex, 0);
}

void Renderer::createGraphicsPipeline() {
	const auto shaderPath = fs::path(SHADER_BINARY_DIR) + SHADER_NAME;
	const auto shaderCode = fs::readFile(shaderPath);
	const auto shaderModule = createShaderModule(shaderCode);

	const vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName = "vertMain"
	};
	const vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eFragment,
		.module = shaderModule,
		.pName = "fragMain"
	};

	vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	auto bindingDescription = Particle::getBindingDescription();
	auto attributeDescriptions = Particle::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		.pVertexAttributeDescriptions = attributeDescriptions.data()
	};

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		.topology = vk::PrimitiveTopology::ePointList,
		.primitiveRestartEnable = vk::False
	};

	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False,
		.lineWidth = 1.0f
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable = vk::False
	};

	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable = vk::True,
		.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
		.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		.colorBlendOp = vk::BlendOp::eAdd,
		.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		.dstAlphaBlendFactor = vk::BlendFactor::eZero,
		.alphaBlendOp = vk::BlendOp::eAdd,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
	mGraphicsPipelineLayout = vk::raii::PipelineLayout(mDevice, pipelineLayoutInfo);

	vk::StructureChain<
		vk::GraphicsPipelineCreateInfo,
		vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		{
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = mGraphicsPipelineLayout,
			.renderPass = nullptr
		},
		{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &mSwapChain.surfaceFormat().format,
		}
	};

	mGraphicsPipeline = vk::raii::Pipeline(
		mDevice,
		nullptr,
		pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void Renderer::createComputePipeline() {
	const auto shaderPath = fs::path(SHADER_BINARY_DIR) + SHADER_NAME;
	const auto shaderCode = fs::readFile(shaderPath);
	const auto shaderModule = createShaderModule(shaderCode);

	const vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eCompute,
		.module = shaderModule,
		.pName = "compMain"
	};
	const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*mComputeDescriptorSetLayout
	};
	mComputePipelineLayout = vk::raii::PipelineLayout(mDevice, pipelineLayoutInfo);

	const vk::ComputePipelineCreateInfo pipelineInfo{
		.stage = computeShaderStageInfo,
		.layout = *mComputePipelineLayout
	};
	mComputePipeline = vk::raii::Pipeline(mDevice, nullptr, pipelineInfo);
}

void Renderer::createCommandPool() {
	const vk::CommandPoolCreateInfo poolInfo{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = mQueueIndex
	};

	mCommandPool = vk::raii::CommandPool(mDevice, poolInfo);
}

void Renderer::createUniformBuffers() {
	mUniformBuffers.clear();

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		mUniformBuffers.emplace_back(
			bufferSize,
			mDevice,
			mPhysicalDevice,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		mUniformBuffers[i].map(bufferSize);
	}
}

void Renderer::createShaderStorageBuffers() {
	const auto particles = Particle::generate(PARTICLE_COUNT);
	constexpr vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

	// Create a staging buffer used to upload data to the gpu
	Buffer stagingBuffer{
		bufferSize,
		mDevice,
		mPhysicalDevice,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
	};

	void* mem = stagingBuffer.map(bufferSize);
	memcpy(mem, particles.data(), bufferSize);
	stagingBuffer.unmap();

	mShaderStorageBuffers.clear();

	// Copy initial particle data to all storage buffers
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		Buffer ssbo{
			bufferSize,
			mDevice,
			mPhysicalDevice,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
			vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		};

		copyBuffer(ssbo, stagingBuffer, bufferSize);

		mShaderStorageBuffers.emplace_back(std::move(ssbo));
	}
}

void Renderer::createDescriptorPool() {
	constexpr std::array<vk::DescriptorPoolSize, 2> poolSizes = {
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2}
	};

	const vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = poolSizes.size(),
		.pPoolSizes = poolSizes.data()
	};

	mDescriptorPool = vk::raii::DescriptorPool(mDevice, poolInfo);
}

void Renderer::createComputeDescriptorSets() {
	std::vector<vk::DescriptorSetLayout> layouts{MAX_FRAMES_IN_FLIGHT, *mComputeDescriptorSetLayout};
	const vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = mDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	mComputeDescriptorSets.clear();
	mComputeDescriptorSets = mDevice.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		vk::DescriptorBufferInfo bufferInfo{
			.buffer = mUniformBuffers[i].getBuffer(),
			.offset = 0,
			.range = mUniformBuffers[i].getSize()
		};

		vk::DescriptorBufferInfo storageBufferInfoLastFrame{
			.buffer = mShaderStorageBuffers[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT].getBuffer(),
			.offset = 0,
			.range = sizeof(Particle) * PARTICLE_COUNT
		};
		vk::DescriptorBufferInfo storageBufferInfoCurrentFrame{
			.buffer = mShaderStorageBuffers[i].getBuffer(),
			.offset = 0,
			.range = sizeof(Particle) * PARTICLE_COUNT
		};
		std::array<vk::WriteDescriptorSet, 3> descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *mComputeDescriptorSets[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pImageInfo = nullptr,
				.pBufferInfo = &bufferInfo,
				.pTexelBufferView = nullptr
			},
			vk::WriteDescriptorSet{
				.dstSet = *mComputeDescriptorSets[i],
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pImageInfo = nullptr,
				.pBufferInfo = &storageBufferInfoLastFrame,
				.pTexelBufferView = nullptr
			},
			vk::WriteDescriptorSet{
				.dstSet = *mComputeDescriptorSets[i],
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pImageInfo = nullptr,
				.pBufferInfo = &storageBufferInfoCurrentFrame,
				.pTexelBufferView = nullptr
			},
		};

		mDevice.updateDescriptorSets(descriptorWrites, {});
	}
}

void Renderer::createComputeDescriptorSetLayout() {
	std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{
		vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute,
		                               nullptr),
		vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute,
		                               nullptr),
		vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute,
		                               nullptr)
	};

	const vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data()
	};

	mComputeDescriptorSetLayout = vk::raii::DescriptorSetLayout(mDevice, layoutInfo);
}

void Renderer::createCommandBuffers() {
	mGraphicsCommandBuffers.clear();
	mComputeCommandBuffers.clear();

	const vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = *mCommandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT
	};

	mGraphicsCommandBuffers = vk::raii::CommandBuffers(mDevice, allocInfo);
	mComputeCommandBuffers = vk::raii::CommandBuffers(mDevice, allocInfo);
}

void Renderer::recordGraphicsCommandBuffer(const uint32_t imageIndex) {
	const auto& commandBuffer = mGraphicsCommandBuffers[mFrameIndex];
	commandBuffer.reset();
	commandBuffer.begin({});

	const auto image = mSwapChain.images()[imageIndex];
	// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
	transitionImageLayout(
		image,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{}, // srcAccessMask (no need to wait for previous operations)
		vk::AccessFlagBits2::eColorAttachmentWrite, // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
		commandBuffer
	);
	constexpr vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = mSwapChain.imageViews()[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	const vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = mSwapChain.extent()},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *mGraphicsPipeline);
	commandBuffer.setViewport(
		0,
		vk::Viewport(
			0.0f,
			0.0f,
			static_cast<float>(mSwapChain.extent().width),
			static_cast<float>(mSwapChain.extent().height),
			0.0f,
			1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), mSwapChain.extent()));
	commandBuffer.bindVertexBuffers(0, {mShaderStorageBuffers[mFrameIndex].getBuffer()}, {0});
	commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
	commandBuffer.endRendering();
	// After rendering, transition the swapchain image to PRESENT_SRC
	transitionImageLayout(
		image,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite, // srcAccessMask
		{}, // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
		vk::PipelineStageFlagBits2::eBottomOfPipe, // dstStage
		commandBuffer
	);
	commandBuffer.end();
}

void Renderer::recordComputeCommandBuffer() {
	const auto& commandBuffer = mComputeCommandBuffers[mFrameIndex];
	commandBuffer.reset();
	commandBuffer.begin({});
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, mComputePipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, mComputePipelineLayout, 0,
	                                 {mComputeDescriptorSets[mFrameIndex]}, {});
	commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);
	commandBuffer.end();
}

void Renderer::createSyncObjects() {
	mFences.clear();

	vk::SemaphoreTypeCreateInfo semaphoreType{.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
	mSemaphore = vk::raii::Semaphore(mDevice, {.pNext = &semaphoreType});
	mTimelineValue = 0;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		vk::FenceCreateInfo fenceInfo{};
		mFences.emplace_back(mDevice, fenceInfo);
	}
}

void Renderer::getPhysicalDevice() {
	const auto physicalDevices = mInstance.enumeratePhysicalDevices();

	if (physicalDevices.empty()) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support!");
	}

	for (auto& phyDevice: physicalDevices) {
		if (checkDeviceSuitable(phyDevice)) {
			mPhysicalDevice = phyDevice;
			break;
		}
	}
}

std::vector<const char*> Renderer::getRequiredInstanceExtensions() {
	uint32_t glfwExtensionCount = 0;
	const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	if (enableValidationLayers) {
		extensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	return extensions;
}

vk::Bool32 Renderer::debugCallback(
	vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	const vk::DebugUtilsMessageTypeFlagsEXT type,
	const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {
	std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

	return vk::False;
}

bool Renderer::checkDeviceSuitable(const vk::raii::PhysicalDevice& phyDevice) {
	// Check if the physicalDevice supports the Vulkan 1.3 API version
	bool supportsVulkan1_3 = phyDevice.getProperties().apiVersion >= vk::ApiVersion13;

	// Check if any of the queue families support graphics operations
	bool supportsGraphics{false};
	for (const auto& qfp: phyDevice.getQueueFamilyProperties()) {
		if (qfp.queueFlags & vk::QueueFlagBits::eGraphics) {
			supportsGraphics = true;
			break;
		}
	}

	// Check if all required physicalDevice extensions are available
	std::unordered_set<std::string_view> availableSet;
	for (const auto& [extensionName, specVersion]: phyDevice.enumerateDeviceExtensionProperties()) {
		availableSet.insert(extensionName);
	}

	bool supportsAllRequiredExtensions{true};
	for (const char* required: deviceExtensions) {
		if (!availableSet.contains(required)) {
			supportsAllRequiredExtensions = false;
			break;
		}
	}

	// Check if the physicalDevice supports the required features (dynamic rendering and extended dynamic state)
	auto features2 = phyDevice.getFeatures2<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

	// Perform the checks with clear boolean logic
	bool supportsSamplerAnisotropy = features2.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy;
	bool supportsShaderDrawParameters = features2.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters;
	bool supportsDynamicRendering = features2.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
	bool supportsSynchronization2 = features2.get<vk::PhysicalDeviceVulkan13Features>().synchronization2;
	bool supportsExtendedDynamicState = features2.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().
			extendedDynamicState;
	bool supportsRequiredFeatures =
			supportsSamplerAnisotropy &&
			supportsShaderDrawParameters &&
			supportsDynamicRendering &&
			supportsSynchronization2 &&
			supportsExtendedDynamicState;

	return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
}

void Renderer::copyBuffer(
	const Buffer& dstBuffer,
	const Buffer& srcBuffer,
	const vk::DeviceSize size) const {
	const vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
	commandCopyBuffer.copyBuffer(&*srcBuffer.getBuffer(), &*dstBuffer.getBuffer(), vk::BufferCopy(0, 0, size));
	endSingleTimeCommands(commandCopyBuffer);
}

void Renderer::updateUniformBuffer(const uint32_t currentImage, const float deltaTime) const {
	UniformBufferObject ubo{};
	ubo.deltaTime = static_cast<float>(deltaTime) * 2.0f;

	memcpy(mUniformBuffers[currentImage].getMapped(), &ubo, sizeof(ubo));
}

vk::raii::ShaderModule Renderer::createShaderModule(const std::vector<char>& code) const {
	const vk::ShaderModuleCreateInfo createInfo{
		.codeSize = code.size() * sizeof(char),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};

	vk::raii::ShaderModule shaderModule{mDevice, createInfo};
	return shaderModule;
}

void Renderer::transitionImageLayout(
	const vk::Image image,
	const vk::ImageLayout oldLayout,
	const vk::ImageLayout newLayout,
	const vk::AccessFlags2 srcAccessMask,
	const vk::AccessFlags2 dstAccessMask,
	const vk::PipelineStageFlags2 srcStageMask,
	const vk::PipelineStageFlags2 dstStageMask,
	const vk::raii::CommandBuffer& commandBuffer) {
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = srcStageMask,
		.srcAccessMask = srcAccessMask,
		.dstStageMask = dstStageMask,
		.dstAccessMask = dstAccessMask,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};

	const vk::DependencyInfo dependencyInfo = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};

	commandBuffer.pipelineBarrier2(dependencyInfo);
}

vk::raii::CommandBuffer Renderer::beginSingleTimeCommands() const {
	const vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = mCommandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};

	vk::raii::CommandBuffer commandBuffer = std::move(mDevice.allocateCommandBuffers(allocInfo).front());
	constexpr vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
	commandBuffer.begin(beginInfo);

	return commandBuffer;
}

void Renderer::endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
	commandBuffer.end();

	const vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};

	mQueue.submit(submitInfo, nullptr);
	mQueue.waitIdle();
}
