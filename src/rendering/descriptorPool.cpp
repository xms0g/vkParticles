#include "descriptorPool.h"

DescriptorPool::DescriptorPool(
	const vk::raii::Device& device,
	const uint32_t poolSizeCount,
	const uint32_t maxSets,
	vk::DescriptorPoolCreateFlagBits poolFlags,
	std::span<DescriptorBinding> bindings) {
	std::vector<vk::DescriptorPoolSize> poolSizes;
	poolSizes.reserve(poolSizeCount);

	for (const auto& [type, count] : bindings) {
		poolSizes.emplace_back(vk::DescriptorPoolSize{type, count});
	}

	const vk::DescriptorPoolCreateInfo poolInfo{
		.flags = poolFlags,
		.maxSets = maxSets,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	mDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}
