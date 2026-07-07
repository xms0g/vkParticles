#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

struct DescriptorBinding {
	vk::DescriptorType type;
	uint32_t count;
};

class DescriptorPool {
public:
	DescriptorPool(
		const vk::raii::Device& device,
		uint32_t poolSizeCount,
		uint32_t maxSets,
		vk::DescriptorPoolCreateFlagBits poolFlags,
		std::span<DescriptorBinding> bindings);


	vk::raii::DescriptorPool& operator*() noexcept { return mDescriptorPool; }
	const vk::raii::DescriptorPool& operator*() const noexcept { return mDescriptorPool; }

private:
	vk::raii::DescriptorPool mDescriptorPool{nullptr};
};
