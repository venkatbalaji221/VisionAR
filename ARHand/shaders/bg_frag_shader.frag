# version 330 core

in vec2 TexCoord;

out vec4 frag_Color;
uniform sampler2D webcam_texture;

void main()
{
    frag_Color = texture(webcam_texture, TexCoord);
}