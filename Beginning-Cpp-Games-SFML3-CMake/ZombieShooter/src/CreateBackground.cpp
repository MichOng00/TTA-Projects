#include "ZombieArena.h"
#include <cstdlib>
#include <ctime>

int createBackground(VertexArray& rVA, IntRect arena)
{
    const int TILE_SIZE = 50;
    const int TILE_TYPES = 3;

    // New constants for triangles
    const int VERTS_IN_TRIANGLE = 3;
    const int TRIANGLES_PER_TILE = 2;
    const int VERTS_PER_TILE = VERTS_IN_TRIANGLE * TRIANGLES_PER_TILE; // 6

    int worldWidth  = arena.size.x / TILE_SIZE;
    int worldHeight = arena.size.y / TILE_SIZE;

    // Use triangles instead of quads
    rVA.setPrimitiveType(PrimitiveType::Triangles);

    // Resize to fit all tiles (each tile uses 6 vertices)
    rVA.resize(worldWidth * worldHeight * VERTS_PER_TILE);

    int currentVertex = 0;

    for (int w = 0; w < worldWidth; w++)
    {
        for (int h = 0; h < worldHeight; h++)
        {
            //-------------------------------------------
            // Compute tile world positions (the 4 corners)
            //-------------------------------------------
            float x = static_cast<float>(w * TILE_SIZE);
            float y = static_cast<float>(h * TILE_SIZE);

            sf::Vector2f topLeft     { x,               y };
            sf::Vector2f topRight    { x + TILE_SIZE,   y };
            sf::Vector2f bottomRight { x + TILE_SIZE,   y + TILE_SIZE };
            sf::Vector2f bottomLeft  { x,               y + TILE_SIZE };

            //-------------------------------------------
            // Compute texture coordinates
            //-------------------------------------------

            int verticalOffset;

            // Border tiles use the "wall" texture (last row of tiles)
            if (h == 0 || h == worldHeight - 1 ||
                w == 0 || w == worldWidth  - 1)
            {
                verticalOffset = TILE_TYPES * TILE_SIZE;
            }
            else
            {
                // Random grass/stone/etc tile
                srand(static_cast<unsigned>(time(0)) + h * w - h);
                int tileType = rand() % TILE_TYPES;
                verticalOffset = tileType * TILE_SIZE;
            }

            sf::Vector2f texTL { 0.f,              (float)verticalOffset };
            sf::Vector2f texTR { (float)TILE_SIZE, (float)verticalOffset };
            sf::Vector2f texBR { (float)TILE_SIZE, (float)verticalOffset + TILE_SIZE };
            sf::Vector2f texBL { 0.f,              (float)verticalOffset + TILE_SIZE };

            //-------------------------------------------
            // Write 2 triangles into the vertex array
            //-------------------------------------------

            // Triangle 1: TL, TR, BR
            rVA[currentVertex + 0].position  = topLeft;
            rVA[currentVertex + 0].texCoords = texTL;

            rVA[currentVertex + 1].position  = topRight;
            rVA[currentVertex + 1].texCoords = texTR;

            rVA[currentVertex + 2].position  = bottomRight;
            rVA[currentVertex + 2].texCoords = texBR;

            // Triangle 2: TL, BR, BL
            rVA[currentVertex + 3].position  = topLeft;
            rVA[currentVertex + 3].texCoords = texTL;

            rVA[currentVertex + 4].position  = bottomRight;
            rVA[currentVertex + 4].texCoords = texBR;

            rVA[currentVertex + 5].position  = bottomLeft;
            rVA[currentVertex + 5].texCoords = texBL;

            currentVertex += VERTS_PER_TILE; // move by 6
        }
    }

    return TILE_SIZE;
}
