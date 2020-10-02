#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <nvrtc.h>

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

bool readFileToString(std::string& sourceString, const std::string& filename)
{
	std::ifstream file(filename.c_str());

	if (file.good())
	{
		std::stringstream buffer;
		buffer << file.rdbuf();
		sourceString = buffer.str();

		return true;
	}

	return false;
}

OptixModule createOptixModule(OptixDeviceContext context)
{
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.numPayloadValues = 1;
	pipelineCompileOptions.numAttributeValues = 2;
#ifdef DEBUG
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	char log[2048];

	OptixModule module = nullptr;
	size_t logSize = sizeof(log);

	std::string filename = "C:\\Users\\agleave\\Documents\\OptiX\\build\\raycaster.ptx";
	std::string ptxString;
	bool readSuccess = readFileToString(ptxString, filename);

	if (!readSuccess)
	{
		std::cerr << "Could not read PTX string from file " << filename << std::endl;
		exit(4);
	}

	OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
		context,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxString.c_str(),
		ptxString.size(),
		log,
		&logSize,
		&module
	));

	return module;
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
		OptixModule optixModule = createOptixModule(optixContext);

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
