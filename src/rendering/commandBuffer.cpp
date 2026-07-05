#include "commandBuffer.h"
#include "commandPool.h"

CommandBuffer::CommandBuffer(const vk::raii::Device& device, const CommandPool& commandPool) {
	const vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = **commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};

	mCommandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
}
