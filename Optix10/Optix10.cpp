#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "macros.h"
#include "Model.h"
#include "Renderer.h"

#pragma comment(lib, "glew32s.lib")
#pragma comment(lib, "opengl32.lib")

static void optixLogCallback(unsigned int level, const char* tag, const char* msg, void*)
{
	std::cout << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << msg << std::endl;
}

OptixDeviceContext initOptix()
{
	CUDA_CHECK(cudaFree(nullptr));
	CUcontext cudaCtx = nullptr;

	OPTIX_CHECK(optixInit());
	
	OptixDeviceContext optixContext = nullptr;

	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &optixLogCallback;
	options.logCallbackLevel = 4;

	OPTIX_CHECK(optixDeviceContextCreate(cudaCtx, &options, &optixContext));

	return optixContext;
}

extern "C" int main(int argc, char** argv)
{
	try
	{
		std::cout << "Initialising OptiX 7..." << std::endl;
		OptixDeviceContext optixContext = initOptix();
		std::cout << "Successfully initialised OptiX!" << std::endl << std::endl;
	
		std::cout << "Initialising renderer..." << std::endl;
		auto renderer = std::make_unique<Renderer>();

		std::cout << "Initialising FBX SDK..." << std::endl;
		
		FbxManager* fbxManager = FbxManager::Create();

		Model* model = new Model("C:\\Users\\agleave\\Documents\\OptiX\\data\\mountain1.fbx");
		model->load(fbxManager, renderer.get());
		OptixTraversableHandle accelStructure = model->buildAccelStructure(optixContext);

		fbxManager->Destroy();

		while (!renderer->shouldClose())
		{
			renderer->renderFrame();
		}
	}
	catch (std::runtime_error& e)
	{
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}

	return 0;
}
