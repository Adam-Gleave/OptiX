#include "Model.h"

#include <iostream>

#define VERTEX_SIZE 3

Model::Model(const std::string& filename) :
	filename(filename)
{
}

void Model::load(FbxManager* fbxManager)
{
	if (!meshVertices.empty())
	{
		for (auto& meshVerticesPtr : meshVertices)
		{
			delete meshVerticesPtr;
		}

		meshVertices.clear();
		meshVertexCounts.clear();
	}

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

	if (meshVertices.empty())
	{
		std::cout << "FBX has no associated meshes" << std::endl;
		exit(-1);
	}
}

void Model::loadVertices(FbxMesh* mesh)
{
	const int vertexCount = mesh->GetControlPointsCount();
	meshVertexCounts.push_back(vertexCount);
	meshVertices.push_back(new float[vertexCount * VERTEX_SIZE]());

	const int polys = mesh->GetPolygonCount();

	for (int polyIndex = 0; polyIndex < polys; polyIndex++)
	{
		const int polyVerts = mesh->GetPolygonSize(polyIndex);

		for (int polyVertIndex = 0; polyVertIndex < polyVerts; polyVertIndex++)
		{
			const int controlPointIndex = mesh->GetPolygonVertex(polyIndex, polyVertIndex);
			const FbxVector4 vertex = mesh->GetControlPointAt(controlPointIndex);

			const int offset = polyVertIndex * VERTEX_SIZE;
			meshVertices.back()[offset+0] = vertex.mData[0];
			meshVertices.back()[offset+1] = vertex.mData[1];
			meshVertices.back()[offset+2] = vertex.mData[2];
		}
	}

	std::cout << "Loaded mesh with " << vertexCount << " vertices" << std::endl;
}

const std::string& Model::getFilename() const
{
	return filename;
}
