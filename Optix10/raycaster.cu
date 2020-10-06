#include <optix.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

#include "raycaster.h"

constexpr int numCasts = 360 * 180;
const int elevationDegrees = 180;
const int elevationLimit = 90;

enum Result
{
	Miss = 0,
	Hit
};

__device__ void computeRay(uint3 idx, float3& direction)
{
	int elevation = idx.x % elevationDegrees - elevationLimit;
	int azimuth = floorf(idx.x / elevationDegrees);

	direction.x = sinf(azimuth) * cosf(elevation);
	direction.y = cosf(azimuth) * cosf(elevation);
	direction.z = sinf(elevation);
}

extern "C" __global__ void __raygen__rg()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	if (idx.x < dim.x && idx.x < numCasts)
	{
		float3 rayOrigin = params.origin;
		float3 rayDirection;
		computeRay(idx, rayDirection);

		unsigned int p0 = Miss;

		OptixTraversableHandle handle = params.handle;

		optixTrace(
			params.handle,
			rayOrigin,
			rayDirection,
			0.0f,
			1000.0f,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,
			1,
			0,
			p0
		);

		params.results[idx.x].x = p0;
	}
}

extern "C" __global__ void __closesthit__ch()
{
	optixSetPayload_0(Hit);
}

extern "C" __global__ void __miss__ms()
{
	optixSetPayload_0(Miss);
}
