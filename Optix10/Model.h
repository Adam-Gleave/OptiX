#pragma once

#include <string>
#include <vector>

#include <fbxsdk.h>

class Model
{
public:
	Model(const std::string& filename);

	void load(FbxManager* fbxManager);
	void printVertexInfo() const;

	const std::string& getFilename() const;

private:
	std::string filename;
	std::vector<FbxMesh*> meshes;
};
