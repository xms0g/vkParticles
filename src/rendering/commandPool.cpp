#include "commandPool.h"

void CommandPool::create(const vk::raii::Device& device, const uint32_t queueIndex) {
	const vk::CommandPoolCreateInfo poolInfo{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = queueIndex
	};

	mCommandPool = vk::raii::CommandPool(device, poolInfo);
}
