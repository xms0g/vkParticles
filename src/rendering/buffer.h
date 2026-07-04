#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

class Buffer {
public:
	Buffer(
		vk::DeviceSize size,
		const vk::raii::Device& device,
		const vk::raii::PhysicalDevice& phyDev,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties);

	[[nodiscard]]
	vk::Buffer getBuffer() const;

	vk::DeviceSize getSize() const;

	[[nodiscard]]
	void* getMapped() const;

	[[nodiscard]]
	void* map(size_t size);

	void unmap() const;

private:
	vk::DeviceSize mSize;
	vk::raii::Buffer mBuffer{nullptr};
	vk::raii::DeviceMemory mBufferMemory{nullptr};
	void* mMappedMemory{nullptr};
};
