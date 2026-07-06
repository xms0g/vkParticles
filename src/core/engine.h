#pragma once

#include <memory>

class Renderer;
class Window;
class Engine {
public:
    Engine();

    ~Engine();

    void run();

private:
	std::unique_ptr<Window> mWindow;
	std::unique_ptr<Renderer> mRenderer;

    double mDeltaTime{0.0};
    double mSecondsPreviousFrame{0};
};
