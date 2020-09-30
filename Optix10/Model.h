#pragma once

#include <fbxsdk.h>

class Renderer;

class Model
{
public:
	Model(const std::string& filename);

	void load(FbxManager* fbxManager, Renderer* renderer);

	const std::string& getFilename() const;
	const unsigned int getVertexBuffer() const;

private:
	void loadVertices(FbxMesh* mesh);
	void createVertexBuffer();

	std::string filename;
	std::vector<float> vertices;
	int vertexCount;

	unsigned int vertexBuffer;
};
