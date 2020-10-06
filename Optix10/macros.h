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

#define OPTIX_CHECK_LOG(call)                                                     \
{                                                                                 \
        OptixResult res = call;                                                   \
        const size_t logSizeReturned = logSize;                                   \
        logSize = sizeof(log);                                                    \
                                                                                  \
        if(res != OPTIX_SUCCESS)                                                  \
        {                                                                         \
            std::cerr << "Optix call '" << #call << "' failed: " __FILE__ ":"     \
                      << __LINE__ << ")\nLog:\n" << log                           \
                      << (logSizeReturned > sizeof(log) ? "<TRUNCATED>" : "")     \
                      << std::endl;                                               \
                                                                                  \
            exit(2);                                                              \
        }                                                                         \
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
                                                                            \
        exit(3);                                                            \
    }                                                                       \
}

#define CUDA_SYNC_CHECK()                                           \
{                                                                   \
    cudaDeviceSynchronize();                                        \
    cudaError_t error = cudaGetLastError();                         \
                                                                    \
    if (error != cudaSuccess)                                       \
    {                                                               \
        std::cerr << "CUDA error on synchronize with error '"       \
                    << cudaGetErrorString(error)                    \
                    << "' (" __FILE__ << ":" << __LINE__ << ")\n";  \
        exit(4);                                                    \
    }                                                               \
}