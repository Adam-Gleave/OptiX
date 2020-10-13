#version 330 core
layout (location = 0) in vec3 pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 position;

void main()
{
    vec4 worldPosition = model * vec4(pos, 1.0f);

    position = worldPosition.xyz;
	gl_Position = projection * view * worldPosition;
}
