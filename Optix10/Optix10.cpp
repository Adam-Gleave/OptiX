#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <gl/glew.h>
#include <gl/GL.h>
#include <GLFW/glfw3.h>

#include "Model.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define OPTIX_CHECK(call)																								\
{																														\
	OptixResult result = call;																							\
																														\
	if (result != OPTIX_SUCCESS)																						\
	{																													\
		std::cerr << "OptiX call " << #call << " failed with code " << result << " at line " << __LINE__ << std::endl;	\
		exit(2);																										\
	}																													\
}																														\

void initOptix()
{
	cudaFree(0);

	int numDevices;
	cudaGetDeviceCount(&numDevices);

	if (numDevices == 0)
	{
		throw std::runtime_error("No CUDA capable devices found!");
	}

	std::cout << "Found: " << numDevices << " CUDA devices." << std::endl;

	OPTIX_CHECK(optixInit());
}

void render()
{
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window;

	if (!glfwInit())
	{
		std::cout << "Failed to initialise GLFW" << std::endl;
		exit(-1);
	}

	window = glfwCreateWindow(640, 480, "Renderer", NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		std::cout << "Failed to initialise window" << std::endl;
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glewExperimental = true;

	if (glewInit() != GLEW_OK)
	{
		glfwTerminate();
		std::cout << "Failed to initialise GLEW" << std::endl;
		exit(-1);
	}

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
}

extern "C" int main(int argc, char** argv)
{
	try
	{
		std::cout << "Initialising OptiX 7..." << std::endl;

		initOptix();

		std::cout << "Successfully initialised OptiX!" << std::endl;
		std::cout << "Done. Exiting." << std::endl;
	
		std::cout << "Initialising FBX SDK..." << std::endl;

		FbxManager* fbxManager = FbxManager::Create();

		auto model = std::make_unique<Model>("C:\\Users\\agleave\\Documents\\OptiX\\data\\mountains.fbx");
		model->load(fbxManager);

		render();
	}
	catch (std::runtime_error& e)
	{
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}

	return 0;
}
