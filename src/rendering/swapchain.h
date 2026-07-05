#pragma once
#include <GLFW/glfw3.h>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

class Swapchain {
public:
	Swapchain() = default;

	vk::raii::SwapchainKHR& native();

	vk::SurfaceFormatKHR& surfaceFormat();

	std::vector<vk::Image>& images();

	std::vector<vk::raii::ImageView>& imageViews();

	vk::Extent2D& extent();

	uint32_t acquireNextImage(const vk::raii::Fence& fence) const;

	void create(
		const vk::raii::SurfaceKHR& surface,
		const vk::raii::Device& device,
		const vk::raii::PhysicalDevice& phyDev,
		GLFWwindow* window);

	void recreate(
		const vk::raii::SurfaceKHR& surface,
		const vk::raii::Device& device,
		const vk::raii::PhysicalDevice& phyDev,
		GLFWwindow* window);

private:
	void createSwapchainImageViews(const vk::raii::Device& device);

	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);

	static vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);

	static uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities);

	vk::raii::SwapchainKHR mSwapChain{nullptr};
	vk::SurfaceFormatKHR mSwapChainSurfaceFormat;
	vk::Extent2D mSwapChainExtent;
	std::vector<vk::Image> mSwapChainImages;
	std::vector<vk::raii::ImageView> mSwapChainImageViews;
};
