#pragma once
#include <array>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "core/vertex.hpp"

struct Particle {
	glm::vec2 position;
	glm::vec2 velocity;

	static VertexLayout layout() {
		return {
			.bindingDescription = {.binding = 0, .stride = sizeof(Particle), .inputRate = vk::VertexInputRate::eVertex},
			.attributeDescriptions = {
				vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
				vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, velocity)),
			}
		};
	}

	static std::vector<Particle> generate(const uint32_t count, const float width, const float height) {
		// Initialize particles
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution dist(0.0f, 1.0f);

		// Initial particle positions on a circle
		std::vector<Particle> particles(count);
		for (auto& [position, velocity]: particles) {
			const float r = 0.25f * sqrtf(dist(gen));
			const float theta = dist(gen) * 2.0f * 3.14159265358979323846f;
			const float x = r * cosf(theta) * height / width;
			const float y = r * sinf(theta);
			position = glm::vec2(x, y);
			velocity = normalize(glm::vec2(x, y)) * 0.5f;
		}

		return particles;
	}
};
