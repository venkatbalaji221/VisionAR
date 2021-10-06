# version 330 core

layout(location = 0) in vec3 position;
layout(location = 1 ) in vec2 texcoord;

out vec2 TexCoord;
uniform mat4 modelview_bg;
uniform mat4 proj_persp_bg;

void main()
{
    gl_Position = proj_persp_bg * modelview_bg * vec4(position, 1.0);
    TexCoord = vec2(texcoord.x, texcoord.y); 
}