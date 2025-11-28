#include "MenuGraphics.h"
#include "MenuUpdate.h"
#include "Graphics.h"

void MenuGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    m_MenuPosition = static_pointer_cast<MenuUpdate>(
        genericUpdate)->getPositionPointer();
    m_GameOver = static_pointer_cast<MenuUpdate>(
        genericUpdate)->getGameOverPointer();
    m_CurrentStatus = *m_GameOver;

    // --- TRIANGLES: 2 triangles = 6 vertices ---
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);

    // Cache the UV base values
    uPos = texCoords.position.x;
    vPos = texCoords.position.y;
    texWidth = texCoords.size.x;
    texHeight = texCoords.size.y;

    // By default, menu is "not game over”
    // So apply the "normal" texture region (single-height)
    float u0 = uPos;
    float v0 = vPos;
    float u1 = uPos + texWidth;
    float v1 = vPos + texHeight;

    // TRIANGLE 1: 0,1,2
    canvas[m_VertexStartIndex + 0].texCoords = { u0, v1 };
    canvas[m_VertexStartIndex + 1].texCoords = { u1, v1 };
    canvas[m_VertexStartIndex + 2].texCoords = { u1, v0 };

    // TRIANGLE 2: 0,2,3
    canvas[m_VertexStartIndex + 3].texCoords = { u0, v1 }; // same as vertex 0
    canvas[m_VertexStartIndex + 4].texCoords = { u1, v0 }; // same as vertex 2
    canvas[m_VertexStartIndex + 5].texCoords = { u0, v0 };
}

void MenuGraphics::draw(VertexArray& canvas)
{
    // Check status change (Game Over toggled)
    if (*m_GameOver && !m_CurrentStatus)
    {
        m_CurrentStatus = *m_GameOver;

        // Switch to “game over” texture (offset down by texHeight)
        float u0 = uPos;
        float u1 = uPos + texWidth;
        float vTop = vPos + texHeight;            // top of lower frame
        float vBottom = vPos + 2 * texHeight;     // bottom of lower frame

        // TRIANGLE 1 (0,1,2)
        canvas[m_VertexStartIndex + 0].texCoords = { u0, vBottom };
        canvas[m_VertexStartIndex + 1].texCoords = { u1, vBottom };
        canvas[m_VertexStartIndex + 2].texCoords = { u1, vTop };

        // TRIANGLE 2 (0,2,3)
        canvas[m_VertexStartIndex + 3].texCoords = { u0, vBottom };
        canvas[m_VertexStartIndex + 4].texCoords = { u1, vTop };
        canvas[m_VertexStartIndex + 5].texCoords = { u0, vTop };
    }
    else if (!*m_GameOver && m_CurrentStatus)
    {
        m_CurrentStatus = *m_GameOver;

        // Switch back to the normal texture
        float u0 = uPos;
        float u1 = uPos + texWidth;
        float vTop = vPos;
        float vBottom = vPos + texHeight;

        // TRIANGLE 1
        canvas[m_VertexStartIndex + 0].texCoords = { u0, vBottom };
        canvas[m_VertexStartIndex + 1].texCoords = { u1, vBottom };
        canvas[m_VertexStartIndex + 2].texCoords = { u1, vTop };

        // TRIANGLE 2
        canvas[m_VertexStartIndex + 3].texCoords = { u0, vBottom };
        canvas[m_VertexStartIndex + 4].texCoords = { u1, vTop };
        canvas[m_VertexStartIndex + 5].texCoords = { u0, vTop };
    }

    // --- Update positions for all 6 vertices ---

    const Vector2f& position = m_MenuPosition->position;
    float w = m_MenuPosition->size.x;
    float h = m_MenuPosition->size.y;

    Vector2f p0 = position;
    Vector2f p1 = position + Vector2f(w, 0);
    Vector2f p2 = position + Vector2f(w, h);
    Vector2f p3 = position + Vector2f(0, h);

    // TRIANGLE 1: 0,1,2
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // TRIANGLE 2: 0,2,3
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;
}
