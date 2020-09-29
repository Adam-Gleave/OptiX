#include "Model.h"

#include <iostream>

Model::Model(const std::string& filename) :
	filename(filename)
{
}

void Model::load(FbxManager* fbxManager)
{
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
						meshes.push_back((FbxMesh*) attribute);
					}
				}
			}
		}
	}

	if (!meshes.size())
	{
		std::cout << "FBX has no associated meshes" << std::endl;
		exit(-1);
	}
}

void Model::printVertexInfo() const
{
	if (meshes.empty())
	{
		std::cout << "Model has not been loaded" << std::endl;
		return;
	}
	
	for (int i = 0; i < meshes.size(); i++)
	{
		FbxMesh* mesh = meshes[i];
		
		const int polys = mesh->GetPolygonCount();
		std::cout << "Mesh " << i << " contains " << polys << " polygons" << std::endl;
	
		for (int j = 0; j < polys; j++)
		{
			for (int k = 0; k < mesh->GetPolygonSize(j); k++)
			{
				const int index = mesh->GetPolygonVertex(j, k);
				const FbxVector4 vertex = mesh->GetControlPointAt(index);

				std::cout << "Poly " << j << " vertex " << k << ": "
					<< vertex.mData[0] << ", "
					<< vertex.mData[1] << ", "
					<< vertex.mData[2] << ", " << std::endl;
			}
		}
	}
}

const std::string& Model::getFilename() const
{
	return filename;
}
