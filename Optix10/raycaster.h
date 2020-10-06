struct ProgramParams
{
	OptixTraversableHandle handle;
	uint1* results;
	float3 origin;
};

extern "C" __constant__ ProgramParams params;

struct RayGenData
{
};

struct MissData
{
};

struct HitGroupData
{
};
