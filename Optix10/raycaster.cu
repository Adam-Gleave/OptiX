#include <optix.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "raycaster.h"

enum Result
{
	Miss = 0,
	Hit
};

static __forceinline__ __device__ void computeRay(uint3 idx, float3& direction)
{
	// TODO
}

extern "C" __global__ void __raygen__rg()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	float3 rayOrigin = params.origin;
	float3 rayDirection;
	computeRay(idx, rayDirection);

	unsigned int p0;

	optixTrace(
		params.handle,
		rayOrigin,
		rayDirection,
		0.0f,
		1000.0f,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		0,
		1,
		0,
		p0
	);

	params.results[idx.x] = static_cast<bool>(p0);
}

extern "C" __global__ void __closesthit__ch()
{
	optixSetPayload_0(Result::Hit);
}

extern "C" __global__ void __miss__ms()
{
	optixSetPayload_0(Result::Miss);
}
