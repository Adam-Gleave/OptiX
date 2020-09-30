#pragma once

#include <fbxsdk.h>

#include "../vendor/glm/glm/glm.hpp"

class Renderer;

class Model
{
public:
	Model(const std::string& filename);

	void load(FbxManager* fbxManager, Renderer* renderer);

	const std::string& getFilename() const;
	const unsigned int getVertexArray() const;
	const unsigned int getVertexCount() const;
	const glm::mat4& getModelMatrix() const;

private:
	void loadVertices(FbxMesh* mesh);
	void createVertexBuffer();

	std::string filename;
	std::vector<float> vertices;
	unsigned int vertexCount;

	unsigned int vertexBuffer;
	unsigned int vertexArray;

	glm::mat4 modelMatrix;
};
