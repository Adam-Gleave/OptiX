#include "stdafx.h"

#include "Renderer.h"

#include "Model.h"

Renderer::Renderer()
{
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	if (!glfwInit())
	{
		std::cout << "Failed to initialise GLFW" << std::endl;
		exit(-1);
	}

	window = glfwCreateWindow(640, 480, "Renderer", NULL, NULL);

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

	auto pathString = std::filesystem::current_path().string();

	auto vertexPath = std::string(pathString).append("\\..\\shaders\\basic.vert");
	auto fragmentPath = std::string(pathString).append("\\..\\shaders\\basic.frag");

	loadShaders(vertexPath.c_str(), fragmentPath.c_str());
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
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void Renderer::addModel(Model* model)
{
	models.push_back(model);
}

bool Renderer::shouldClose() const
{
	return glfwWindowShouldClose(window);
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

	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, logSize, NULL, log);
		std::cout << "Error compiling fragment shader" << std::endl;
		std::cout << log << std::endl;
		exit(-1);
	}

	unsigned int program;
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
