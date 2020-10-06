#pragma once

#include <fbxsdk.h>
#include <optix_types.h>

#include "../thirdparty/glm/glm/glm.hpp"

class Renderer;

class Model
{
public:
	Model(const std::string& filename);

	void load(FbxManager* fbxManager, Renderer* renderer);
	OptixTraversableHandle buildAccelStructure(const OptixDeviceContext& context);

	const std::string& getFilename() const;
	const unsigned int getVertexArray() const;
	const unsigned int getVertexCount() const;
	const glm::mat4& getModelMatrix() const;
	const glm::vec3& getColor() const;

private:
	void loadVertices(FbxMesh* mesh);
	void createVertexBuffer();

	std::string filename;
	std::vector<float> vertices;

	unsigned int vertexBuffer;
	unsigned int vertexArray;

	glm::mat4 modelMatrix;
	glm::vec3 color;
};
