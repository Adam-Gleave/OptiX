#pragma once

#include <GLFW/glfw3.h>

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

	GLFWwindow* window;
	std::vector<Model*> models;
};
