#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

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

extern "C" int main(int argc, char** argv)
{
	try
	{
		std::cout << "Initialising OptiX 7..." << std::endl;

		initOptix();

		std::cout << "Successfully initialised OptiX!" << std::endl;
		std::cout << "Done. Exiting." << std::endl;
	}
	catch (std::runtime_error& e)
	{
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}

	return 0;
}
