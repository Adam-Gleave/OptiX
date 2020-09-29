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
	void loadVertices(FbxMesh* mesh);

	std::string filename;
	std::vector<float*> meshVertices;
	std::vector<int> meshVertexCounts;
};
