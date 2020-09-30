#include "stdafx.h"

#include "Model.h"

#include "Renderer.h"

const int VERTEX_SIZE = 3;

Model::Model(const std::string& filename) :
	filename(filename),
	vertexCount(0),
	vertexBuffer(0)
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

	createVertexBuffer();

	if (!vertexCount)
	{
		std::cout << "FBX has no associated meshes" << std::endl;
		exit(-1);
	}

	renderer->addModel(std::move(this));
}

void Model::loadVertices(FbxMesh* mesh)
{
	const int meshVertexCount = mesh->GetControlPointsCount();
	vertexCount += meshVertexCount;

	const int polys = mesh->GetPolygonCount();
	int offset = 0;

	for (int polyIndex = 0; polyIndex < polys; polyIndex++)
	{
		const int polyVerts = mesh->GetPolygonSize(polyIndex);

		for (int polyVertIndex = 0; polyVertIndex < polyVerts; polyVertIndex++)
		{
			const int controlPointIndex = mesh->GetPolygonVertex(polyIndex, polyVertIndex);
			const FbxVector4 vertex = mesh->GetControlPointAt(controlPointIndex);

			vertices.push_back(static_cast<float>(vertex.mData[0]));
			vertices.push_back(static_cast<float>(vertex.mData[1]));
			vertices.push_back(static_cast<float>(vertex.mData[2]));
			offset += VERTEX_SIZE;
		}
	}

	std::cout << "Loaded mesh with " << meshVertexCount << " vertices" << std::endl;
}

void Model::createVertexBuffer()
{
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * VERTEX_SIZE, vertices.data(), GL_STATIC_DRAW);
}

const std::string& Model::getFilename() const
{
	return filename;
}

const unsigned int Model::getVertexBuffer() const
{
	return vertexBuffer;
}
