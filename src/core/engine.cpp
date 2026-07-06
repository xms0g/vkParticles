#include "engine.h"
#include "window.h"
#include "../config/config.hpp"
#include "../rendering/renderer.h"

Engine::Engine()
	: mWindow(std::make_unique<Window>()),
	  mRenderer(std::make_unique<Renderer>(*mWindow)) {
	try {
		mWindow->init("Vulkan Particles", WIDTH, HEIGHT);
		mRenderer->init();
	} catch (const std::runtime_error& e) {
		throw std::runtime_error(e.what());
	}
}

Engine::~Engine() = default;

void Engine::run() {
	while (!mWindow->shouldClose()) {
		glfwPollEvents();

		const double currentTime = glfwGetTime();
		mDeltaTime = static_cast<float>(currentTime - mSecondsPreviousFrame);
		mSecondsPreviousFrame = currentTime;

		mWindow->updateFpsCounter(mDeltaTime);
		mRenderer->render(mDeltaTime);
	}
	mRenderer->waitIdle();
}
