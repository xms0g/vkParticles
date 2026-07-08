#pragma once
#include <vector>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

class DescriptorSetLayout {
public:
	explicit DescriptorSetLayout(const vk::raii::Device& device);

	DescriptorSetLayout& addBinding(
		uint32_t binding,
		vk::DescriptorType type,
		uint32_t count = 1,
		vk::ShaderStageFlags stageFlags = vk::ShaderStageFlagBits::eAll);

	void build();

	vk::raii::DescriptorSetLayout& operator*() noexcept { return mDescriptorSetLayout; }
	const vk::raii::DescriptorSetLayout& operator*() const noexcept { return mDescriptorSetLayout; }

private:
	std::vector<vk::DescriptorSetLayoutBinding> mBindings;
	const vk::raii::Device& mDevice;
	vk::raii::DescriptorSetLayout mDescriptorSetLayout{nullptr};
};
