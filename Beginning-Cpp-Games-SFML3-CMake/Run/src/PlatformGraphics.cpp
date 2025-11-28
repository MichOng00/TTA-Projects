#include "PlatformGraphics.h"
#include "PlatformUpdate.h"
#include "Graphics.h"

void PlatformGraphics::draw(VertexArray& canvas)
{
    const Vector2f& position = m_Position->position;
    const Vector2f& scale = m_Position->size;

    Vector2f p0 = position;
    Vector2f p1 = position + Vector2f(scale.x, 0);
    Vector2f p2 = position + scale;
    Vector2f p3 = position + Vector2f(0, scale.y);

    // Triangle 1: (0,1,2)
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // Triangle 2: (0,2,3)
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;
}

void PlatformGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    shared_ptr<PlatformUpdate> platformUpdate =
        static_pointer_cast<PlatformUpdate>(genericUpdate);

    m_Position = platformUpdate->getPositionPointer();

    // Reserve 6 vertices for two triangles
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);

    const int uPos = texCoords.position.x;
    const int vPos = texCoords.position.y;
    const int texWidth = texCoords.size.x;
    const int texHeight = texCoords.size.y;

    float u0 = uPos;
    float v0 = vPos;
    float u1 = uPos + texWidth;
    float v1 = vPos + texHeight;

    // -------- Triangle 1: 0, 1, 2 --------
    canvas[m_VertexStartIndex + 0].texCoords = { u0, v0 };
    canvas[m_VertexStartIndex + 1].texCoords = { u1, v0 };
    canvas[m_VertexStartIndex + 2].texCoords = { u1, v1 };

    // -------- Triangle 2: 0, 2, 3 --------
    canvas[m_VertexStartIndex + 3].texCoords = { u0, v0 };
    canvas[m_VertexStartIndex + 4].texCoords = { u1, v1 };
    canvas[m_VertexStartIndex + 5].texCoords = { u0, v1 };
}
