#include <optix.h>

#include "raycaster.h"

#include <optix_device.h>

#define HIT 1

extern "C"
{
	__constant__ ProgramParams params;
}

static __forceinline__ __device__ void computeRay(uint1 idx, float3& direction)
{
	// TODO
}

extern "C" __global__ void __raygen__rg()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	float3 rayOrigin, rayDirection;

	unsigned int p0, p1, p2;

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
}

extern "C" __global__ void __closesthit__ch()
{
	optixSetPayload_0(HIT);
}
