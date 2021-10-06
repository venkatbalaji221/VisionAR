# version 330 core

in vec2 TexCoord;

out vec4 frag_Color;
uniform sampler2D crate_texture;
uniform sampler2D baboon_texture;

void main()
{
    frag_Color = mix(texture(crate_texture, TexCoord),texture(baboon_texture, TexCoord), 0.5);
}