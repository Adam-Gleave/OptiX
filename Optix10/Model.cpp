#include "stdafx.h"
#include "Model.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_host.h>

#include "macros.h"
#include "Renderer.h"

const int VERTEX_SIZE = 3;

Model::Model(const std::string& filename) :
	filename(filename),
	vertexBuffer(0),
	modelMatrix(1.0f),
	color(0.7f, 0.7f, 0.7f)
{
}

void Model::load(FbxManager* fbxManager, Renderer* renderer)
{
	vertices.clear();

	FbxIOSettings* fbxIOSettings = FbxIOSettings::Create(fbxManager, IOSROOT);
	FbxImporter* fbxImporter = FbxImporter::Create(fbxManager, "");

	if (!fbxImporter->Initialize(filename.c_str(), -1, fbxManager->GetIOSettings()))
	{
		std::cout << "Failed to initialise FBX importer" << std::endl;
		std::cout << "Error returned: " << fbxImporter->GetStatus().GetErrorString() << std::endl;
		exit(-1);
	}

	FbxScene* scene = FbxScene::Create(fbxManager, "scene");
	fbxImporter->Import(scene);

	FbxGeometryConverter converter(fbxManager);
	converter.Triangulate(scene, true);

	fbxImporter->Destroy();

	FbxNode* rootNode = scene->GetRootNode();

	if (rootNode)
	{
		for (int i = 0; i < rootNode->GetChildCount(); i++)
		{
			FbxNode* node = rootNode->GetChild(i);
			
			if (node)
			{
				for (int j = 0; j < node->GetNodeAttributeCount(); j++)
				{
					FbxNodeAttribute* attribute = node->GetNodeAttributeByIndex(i);

					if (attribute && attribute->GetAttributeType() == FbxNodeAttribute::eMesh)
					{
						loadVertices((FbxMesh*)attribute);
					}
				}
			}
		}
	}

	scene->Destroy();

	if (!vertices.size())
	{
		std::cout << "FBX has no associated meshes" << std::endl;
		exit(-1);
	}

	createVertexBuffer();
	renderer->addModel(std::move(this));
}

void Model::loadVertices(FbxMesh* mesh)
{
	FbxVector4* meshVertices = mesh->GetControlPoints();

	for (int i = 0; i < mesh->GetPolygonCount(); i++)
	{
		int vertsPerPoly = mesh->GetPolygonSize(i);

		for (int j = 0; j < vertsPerPoly; j++)
		{
			int vertID = mesh->GetPolygonVertex(i, j);
			FbxVector4 vertex = meshVertices[vertID];

			vertices.push_back(static_cast<float>(vertex.mData[0]));
			vertices.push_back(static_cast<float>(vertex.mData[1]));
			vertices.push_back(static_cast<float>(vertex.mData[2]));
		}
	}

	const int meshVertexCount = mesh->GetControlPointsCount();
	std::cout << "Loaded mesh with " << meshVertexCount << " vertices" << std::endl;
}

void Model::createVertexBuffer()
{
	glGenBuffers(1, &vertexBuffer);
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}

OptixTraversableHandle Model::buildAccelStructure(const OptixDeviceContext& context)
{
	OptixTraversableHandle accelStructureHandle;

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	const size_t verticesSize = vertices.size() * sizeof(float);

	CUdeviceptr deviceVertexBuffer = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVertexBuffer), verticesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(deviceVertexBuffer), vertices.data(), verticesSize, cudaMemcpyHostToDevice));

	OptixBuildInput triangleInput = {};
	const uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
	triangleInput.triangleArray.flags = triangleInputFlags;
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.numVertices = static_cast<uint32_t>(vertices.size() / 3);
	triangleInput.triangleArray.vertexBuffers = &deviceVertexBuffer;
	triangleInput.triangleArray.numSbtRecords = 1;

	OptixAccelBufferSizes bufferSizes;

	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &triangleInput, 1, &bufferSizes));
	
	CUdeviceptr tempBuffer = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));

	CUdeviceptr outputBuffer = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), bufferSizes.outputSizeInBytes));

	OPTIX_CHECK(optixAccelBuild(
		context,
		0,
		&accelOptions,
		&triangleInput,
		1,
		tempBuffer,
		bufferSizes.tempSizeInBytes,
		outputBuffer,
		bufferSizes.outputSizeInBytes,
		&accelStructureHandle,
		nullptr,
		0
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(deviceVertexBuffer)));

	return accelStructureHandle;
}

const std::string& Model::getFilename() const
{
	return filename;
}

const unsigned int Model::getVertexArray() const
{
	return vertexArray;
}

const unsigned int Model::getVertexCount() const
{
	return vertices.size();
}

const glm::mat4& Model::getModelMatrix() const
{
	return modelMatrix;
}

const glm::vec3& Model::getColor() const
{
	return color;
}
