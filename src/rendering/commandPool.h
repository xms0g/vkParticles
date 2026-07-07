#pragma once
#include <cstdint>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

class CommandPool {
public:
	CommandPool(const vk::raii::Device& device, const uint32_t queueIndex);

	vk::raii::CommandPool& operator*() noexcept { return mCommandPool; }
	const vk::raii::CommandPool& operator*() const noexcept { return mCommandPool; }

private:
	vk::raii::CommandPool mCommandPool{nullptr};
};
