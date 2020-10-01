#include "stdafx.h"

#include "Camera.h"

#include "../vendor/glm/glm/glm.hpp"
#include "../vendor/glm/glm/gtc/matrix_transform.hpp"
#include "../vendor/glm/glm/gtc/type_ptr.hpp"

WindowParams::WindowParams() :
	height(768.0f),
	width(1024.0f)
{
}

Camera::Camera() :
	near(0.1f),
	far(1000.0f),
	fov(45.0f),
	position(64.0f, 64.0f, 64.0f),
	up(0.0f, 1.0f, 0.0f),
	speed(20.0f),
	sensitivity(0.1f),
	pitch(-45.0f),
	yaw(225.0f)
{
	direction = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - position);
	updateViewMatrix();
}

void Camera::processMovement(CameraMovement movement, float factor)
{
	switch (movement)
	{
	case CameraMovement::FORWARD:
	{
		position += speed * factor * direction;
		break;
	}
	case CameraMovement::BACK:
	{
		position -= speed * factor * direction;
		break;
	}
	case CameraMovement::RIGHT:
	{
		position += glm::normalize(glm::cross(direction, up)) * speed * factor;
		break;
	}
	case CameraMovement::LEFT:
	{
		position -= glm::normalize(glm::cross(direction, up)) * speed * factor;
		break;
	}
	case CameraMovement::YAW:
	{
		yaw += factor * sensitivity;
		break;
	}
	case CameraMovement::PITCH:
	{
		pitch += factor * sensitivity;

		if (pitch > 89.9f)
		{
			pitch = 89.9f;
		}
		else if (pitch < -89.9f)
		{
			pitch = -89.9f;
		}

		break;
	}
	}

	updateViewMatrix();
}

void Camera::updateViewMatrix()
{
	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

	viewMatrix = glm::lookAt(position, position + direction, up);
}

const glm::mat4& Camera::getViewMatrix() const
{
	return viewMatrix;
}

const glm::mat4& Camera::getProjectionMatrix(WindowParams* windowParams) const
{
	return glm::perspective(
		glm::radians(fov),
		windowParams->width / windowParams->height,
		near,
		far
	);
}
