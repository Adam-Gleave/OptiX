#include "stdafx.h"

#include "Renderer.h"

#include "Model.h"
#include "../thirdparty/glm/glm/glm.hpp"
#include "../thirdparty/glm/glm/gtc/matrix_transform.hpp"
#include "../thirdparty/glm/glm/gtc/type_ptr.hpp"

Renderer::Renderer() :
	deltaT(0.0f),
	prevT(0.0f),
	prevX(windowParams.width / 2.0f),
	prevY(windowParams.height / 2.0f),
	toClose(false)
{
	glfwWindowHint(GLFW_SAMPLES, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	if (!glfwInit())
	{
		std::cout << "Failed to initialise GLFW" << std::endl;
		exit(-1);
	}

	window = glfwCreateWindow(windowParams.width, windowParams.height, "Renderer", NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		std::cout << "Failed to initialise window" << std::endl;
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glewExperimental = true;

	if (glewInit() != GLEW_OK)
	{
		glfwTerminate();
		std::cout << "Failed to initialise GLEW" << std::endl;
		exit(-1);
	}

	std::string vertexPath = "C:\\Users\\agleave\\Documents\\OptiX\\shaders\\basic.vert";
	std::string fragmentPath = "C:\\Users\\agleave\\Documents\\OptiX\\shaders\\basic.frag";

	loadShaders(vertexPath.c_str(), fragmentPath.c_str());
	initCamera();

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPos(window, windowParams.width / 2.0f, windowParams.height / 2.0f);
	glEnable(GL_DEPTH_TEST);
}

Renderer::~Renderer()
{
	for (int i = 0; i < models.size(); i++)
	{
		delete models[i];
	}

	models.clear();
	glfwTerminate();
}

void Renderer::renderFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	processKeyInput();
	processMouseInput();

	for (const auto& model : models)
	{
		int viewUniformLoc = glGetUniformLocation(program, "view");
		glUniformMatrix4fv(viewUniformLoc, 1, GL_FALSE, glm::value_ptr(camera.getViewMatrix()));

		int modelUniformLoc = glGetUniformLocation(program, "model");
		glUniformMatrix4fv(modelUniformLoc, 1, GL_FALSE, glm::value_ptr(model->getModelMatrix()));

		int objectColorLoc = glGetUniformLocation(program, "objectColor");
		glUniform3fv(objectColorLoc, 1, glm::value_ptr(model->getColor()));

		glBindVertexArray(model->getVertexArray());
		glDrawArrays(GL_TRIANGLES, 0, model->getVertexCount());
	}

	glfwSwapBuffers(window);
	glfwPollEvents();
}

void Renderer::processKeyInput()
{
	const float currT = glfwGetTime();
	deltaT = currT - prevT;
	prevT = currT;
	
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		toClose = true;
		return;
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera.processMovement(CameraMovement::FORWARD, deltaT);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera.processMovement(CameraMovement::BACK, deltaT);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera.processMovement(CameraMovement::LEFT, deltaT);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera.processMovement(CameraMovement::RIGHT, deltaT);
	}
}

void Renderer::processMouseInput()
{
	double currX;
	double currY;
	glfwGetCursorPos(window, &currX, &currY);

	if (currX != prevX)
	{
		const float xOffset = currX - prevX;
		camera.processMovement(CameraMovement::YAW, xOffset);
		prevX = currX;
	}

	if (currY != prevY)
	{
		const float yOffset = prevY - currY;
		camera.processMovement(CameraMovement::PITCH, yOffset);
		prevY = currY;
	}
}

void Renderer::addModel(Model* model)
{
	models.push_back(model);
}

bool Renderer::shouldClose() const
{
	return glfwWindowShouldClose(window) || toClose;
}

void Renderer::loadShaders(const char* vert, const char* frag)
{
	const std::string vertexSourceString = loadShaderSource(vert);
	const std::string fragmentSourceString = loadShaderSource(frag);

	const char* vertexSource = vertexSourceString.c_str();
	const char* fragmentSource = fragmentSourceString.c_str();

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	const int logSize = 512;
	int success;
	char log[logSize];

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertexShader, logSize, NULL, log);
		std::cout << "Error compiling vertex shader " << std::endl;
		std::cout << log << std::endl;
		exit(-1);
	}

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

	std::cout << vertexSource << std::endl;

	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, logSize, NULL, log);
		std::cout << "Error compiling fragment shader" << std::endl;
		std::cout << log << std::endl;
		exit(-1);
	}

	program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &success);
	
	if (!success)
	{
		glGetProgramInfoLog(program, logSize, NULL, log);
		std::cout << "Error linking shader program" << std::endl;
		std::cout << log << std::endl;
		exit(-1);
	}

	std::cout << fragmentSource << std::endl;

	glUseProgram(program);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

const std::string Renderer::loadShaderSource(const char* filename)
{
	std::string source;
	std::string line;
	std::ifstream file(filename);

	if (file.is_open())
	{
		while (std::getline(file, line))
		{
			source.append(line);
			source.append("\n");
		}

		file.close();
	}
	else
	{
		std::cout << "Unable to open shader file " << filename << std::endl;
		exit(-1);
	}

	return source;
}

void Renderer::initCamera()
{
	glm::mat4 projection = camera.getProjectionMatrix(&windowParams);
	int projUniformLoc = glGetUniformLocation(program, "projection");
	glUniformMatrix4fv(projUniformLoc, 1, GL_FALSE, glm::value_ptr(projection));

	glm::vec3 lightPosition = glm::vec3(128.0f, 128.0f, -128.0f);
	int lightPositionLoc = glGetUniformLocation(program, "lightPosition");
	glUniform3fv(lightPositionLoc, 1, glm::value_ptr(lightPosition));

	glm::vec3 lightColor = glm::vec3(0.95f, 0.95f, 0.95f);
	int lightColorLoc = glGetUniformLocation(program, "lightColor");
	glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));

	glm::vec3 ambientColor = glm::vec3(0.2f, 0.2f, 0.2f);
	int ambientLoc = glGetUniformLocation(program, "ambientColor");
	glUniform3fv(ambientLoc, 1, glm::value_ptr(ambientColor));
}
