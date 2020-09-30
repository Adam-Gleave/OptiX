#pragma once

#include <GLFW/glfw3.h>

#include "Camera.h"

class Model;

class Renderer
{
public:
	Renderer();
	~Renderer();

	void renderFrame();
	void addModel(Model* model);

	bool shouldClose() const;

private:
	void loadShaders(const char* vert, const char* frag);
	const std::string loadShaderSource(const char* filename);

	void initCamera();

	void processKeyInput();
	void processMouseInput();

	std::vector<Model*> models;

	GLFWwindow* window;
	WindowParams windowParams;
	bool toClose;

	Camera camera;
	float deltaT;
	float prevT;
	float prevX;
	float prevY;

	unsigned int program;
};
