#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <nvrtc.h>

#include "macros.h"
#include "Model.h"
#include "Renderer.h"

#include "raycaster.h";

#pragma comment(lib, "glew32s.lib")
#pragma comment(lib, "opengl32.lib")

const float3 origin = { 0.0f, 0.0f, 32.0f };
constexpr int numCasts = 360 * 180;

template<typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

enum ProgramGroupIndex {
	RaygenGroup = 0,
	HitGroup,
	MissGroup,
	ProgramGroupCount
};

using OptixProgramGroups = std::array<OptixProgramGroup, ProgramGroupCount>;

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

OptixPipelineCompileOptions createOptixPipelineCompileOptions()
{
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.numPayloadValues = 1;
	pipelineCompileOptions.numAttributeValues = 2;
//#ifdef DEBUG
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
//#else
//	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
//#endif
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	return pipelineCompileOptions;
}

OptixModule createOptixModule(const OptixDeviceContext& context, const OptixPipelineCompileOptions& pipelineCompileOptions)
{
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

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

OptixProgramGroups createOptixProgramGroups(const OptixDeviceContext& context, const OptixModule& optixModule)
{
	OptixProgramGroup raygenGroup = nullptr;
	OptixProgramGroup missGroup = nullptr;
	OptixProgramGroup hitGroup = nullptr;

	OptixProgramGroupOptions programGroupOptions = {};

	char log[2048];
	size_t logSize = sizeof(log);

	OptixProgramGroupDesc raygenGroupDesc = {};
	raygenGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenGroupDesc.raygen.module = optixModule;
	raygenGroupDesc.raygen.entryFunctionName = "__raygen__rg";

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&raygenGroupDesc,
		1,
		&programGroupOptions,
		log,
		&logSize,
		&raygenGroup
	));

	OptixProgramGroupDesc missGroupDesc = {};
	missGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missGroupDesc.miss.module = optixModule;
	missGroupDesc.miss.entryFunctionName = "__miss__ms";

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&raygenGroupDesc,
		1,
		&programGroupOptions,
		log,
		&logSize,
		&missGroup
	));

	OptixProgramGroupDesc hitGroupDesc = {};
	hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitGroupDesc.hitgroup.moduleCH = optixModule;
	hitGroupDesc.hitgroup.moduleAH = nullptr;
	hitGroupDesc.hitgroup.moduleIS = nullptr;
	hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
	hitGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
	hitGroupDesc.hitgroup.entryFunctionNameIS = nullptr;

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&hitGroupDesc,
		1,
		&programGroupOptions,
		log,
		&logSize,
		&hitGroup
	));

	return OptixProgramGroups { raygenGroup, missGroup, hitGroup };
}

OptixPipeline createOptixPipeline(const OptixDeviceContext& context, const OptixProgramGroups& programGroups, const OptixPipelineCompileOptions& pipelineCompileOptions)
{
	OptixPipeline pipeline = nullptr;

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 1;
//#ifdef DEBUG
	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//#else
//	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
//#endif

	char log[2048];
	size_t logSize = sizeof(log);

	std::cout << "Program groups size: " << programGroups.size() << std::endl;

	OPTIX_CHECK_LOG(optixPipelineCreate(
		context,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programGroups.data(),
		programGroups.size(),
		log,
		&logSize,
		&pipeline
	));

	OptixStackSizes stackSizes = {};

	for (const auto& programGroup : programGroups)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, &stackSizes));
	}

	uint32_t directCallableStackSizeFromTraversal;
	uint32_t directCallableStackSizeFromState;
	uint32_t continuationStackSize;

	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stackSizes,
		1,
		0,
		0,
		&directCallableStackSizeFromTraversal,
		&directCallableStackSizeFromState,
		&continuationStackSize
	));

	OPTIX_CHECK(optixPipelineSetStackSize(
		pipeline,
		directCallableStackSizeFromTraversal,
		directCallableStackSizeFromState,
		continuationStackSize,
		1
	));

	return pipeline;
}

OptixShaderBindingTable createOptixSBT(const OptixProgramGroups& programGroups)
{
	OptixShaderBindingTable sbt = {};

	CUdeviceptr raygenRecord;
	const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));
	RayGenSbtRecord raygenSbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[RaygenGroup], &raygenSbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygenRecord), &raygenSbt, raygenRecordSize, cudaMemcpyHostToDevice));

	CUdeviceptr missRecord;
	const size_t missRecordSize = sizeof(MissSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));
	RayGenSbtRecord missSbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[MissGroup], &missSbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(missRecord), &missSbt, missRecordSize, cudaMemcpyHostToDevice));

	CUdeviceptr hitGroupRecord;
	const size_t hitGroupRecordSize = sizeof(HitGroupSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitGroupRecord), hitGroupRecordSize));
	RayGenSbtRecord hitGroupSbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[HitGroup], &hitGroupSbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitGroupRecord), &hitGroupSbt, hitGroupRecordSize, cudaMemcpyHostToDevice));

	sbt.raygenRecord = raygenRecord;
	sbt.missRecordBase = missRecord;
	sbt.missRecordCount = 1;
	sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	sbt.hitgroupRecordBase = hitGroupRecord;
	sbt.hitgroupRecordCount = 1;
	sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);

	return sbt;
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
		fbxManager->Destroy();

		OptixTraversableHandle accelStructure = model->buildAccelStructure(optixContext);
		OptixPipelineCompileOptions optixPipelineCompileOptions = createOptixPipelineCompileOptions();
		OptixModule optixModule = createOptixModule(optixContext, optixPipelineCompileOptions);
		OptixProgramGroups optixProgramGroups = createOptixProgramGroups(optixContext, optixModule);
		OptixPipeline optixPipeline = createOptixPipeline(optixContext, optixProgramGroups, optixPipelineCompileOptions);
		OptixShaderBindingTable sbt = createOptixSBT(optixProgramGroups);

		// Launch OptiX kernel
		{
			CUstream stream;
			CUDA_CHECK(cudaStreamCreate(&stream));

			uint1* deviceOutput = nullptr;
			CUDA_CHECK(cudaSetDevice(0));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(deviceOutput)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), numCasts * sizeof(uint1)));

			ProgramParams params;
			params.handle = accelStructure;
			params.origin = origin;
			params.results = deviceOutput;

			CUdeviceptr deviceParams;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceParams), sizeof(ProgramParams)));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(deviceParams), &params, sizeof(params), cudaMemcpyHostToDevice));

			OPTIX_CHECK(optixLaunch(optixPipeline, stream, deviceParams, sizeof(ProgramParams), &sbt, numCasts, 1, 1));

			std::vector<uint1> hostOutput;
			hostOutput.resize(numCasts);
			CUDA_CHECK(cudaMemcpy(
				static_cast<void*>(hostOutput.data()), 
				deviceOutput, 
				numCasts * sizeof(uint1), 
				cudaMemcpyDeviceToHost
			));

			CUDA_SYNC_CHECK();
			CUDA_CHECK(cudaStreamSynchronize(stream));
		
			std::cout << std::endl;
			for (const auto& value : hostOutput)
			{
				std::cout << value.x << ", ";
			}
			std::cout << std::endl;
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));

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
