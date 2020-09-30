#version 330 core
in vec3 position;

uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 objectColor;

out vec4 FragColor;

void main()
{
    vec3 x = dFdx(position);
    vec3 y = dFdy(position);
    vec3 normal = normalize(cross(x, y));
    
    vec3 lightDirection = normalize(lightPosition - position);
    
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diffuseFactor * lightColor;
    
    vec3 color = (ambientColor + diffuse) * objectColor;
    
    FragColor = vec4(color, 1.0f);
}