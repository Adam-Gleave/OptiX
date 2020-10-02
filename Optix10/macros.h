#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix_types.h>

#define OPTIX_CHECK(call)																								\
{																														\
	OptixResult result = call;																							\
																														\
	if (result != OPTIX_SUCCESS)																						\
	{																													\
		std::cerr << "OptiX call " << #call << " failed with code " << result << " at line " << __LINE__ << std::endl;	\
		exit(2);																										\
	}																													\
}

#define CUDA_CHECK(call)                                                    \
{                                                                           \
    cudaError_t error = call;                                               \
                                                                            \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        std::cerr << "CUDA call (" << #call << ") failed with error: '"     \
                  << cudaGetErrorString(error)                              \
                  << "' (" __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(3);                                                            \
    }                                                                       \
}