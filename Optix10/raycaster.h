struct ProgramParams
{
	OptixTraversableHandle handle;
	bool* results;
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
