#include "buffer.h"
#include "memory.hpp"

Buffer::Buffer(
	const vk::DeviceSize size,
	const vk::raii::Device& device,
	const vk::raii::PhysicalDevice& phyDev,
	const vk::BufferUsageFlags usage,
	const vk::MemoryPropertyFlags properties)
	: mSize(size) {
	const vk::BufferCreateInfo bufferInfo{
		.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive
	};

	mBuffer = vk::raii::Buffer(device, bufferInfo);

	const vk::MemoryRequirements memRequirements = mBuffer.getMemoryRequirements();
	const vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = memory::findMemoryType(memRequirements.memoryTypeBits, properties, phyDev)
	};

	mBufferMemory = vk::raii::DeviceMemory(device, allocInfo);
	mBuffer.bindMemory(mBufferMemory, 0);
}

vk::Buffer Buffer::getBuffer() const {
	return mBuffer;
}

vk::DeviceSize Buffer::getSize() const {
	return mSize;
}

void* Buffer::getMapped() const {
	return mMappedMemory;
}

void* Buffer::map(const size_t size) {
	mMappedMemory = mBufferMemory.mapMemory(0, size);
	return mMappedMemory;
}

void Buffer::unmap() const {
	mBufferMemory.unmapMemory();
}
