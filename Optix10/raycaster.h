#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

struct ProgramParams
{
	OptixTraversableHandle handle;
	float3 origin;
};

struct RayGenData
{
};

struct HitGroupData
{
};
