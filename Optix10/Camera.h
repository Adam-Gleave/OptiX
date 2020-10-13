#pragma once

#undef near
#undef far

#include "../thirdparty/glm/glm/glm.hpp"

enum class CameraMovement
{
	RIGHT,
	LEFT,
	FORWARD,
	BACK,
	YAW,
	PITCH
};

struct WindowParams
{
	float height;
	float width;

	WindowParams();
};

class Camera
{
public:
	Camera();

	void processMovement(CameraMovement movement, float deltaT);

	void updateViewMatrix();

	const glm::mat4& getViewMatrix() const;
	const glm::mat4 getProjectionMatrix(WindowParams* windowParams) const;

private:
	float near;
	float far;
	float fov;
	float speed;
	float sensitivity;

	float pitch;
	float yaw;

	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;

	glm::mat4 viewMatrix;
};
