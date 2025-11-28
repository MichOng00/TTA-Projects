#include "FireballGraphics.h"
#include "Animator.h"
#include "FireballUpdate.h"

void FireballGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    shared_ptr<FireballUpdate> fu =
        static_pointer_cast<FireballUpdate>(genericUpdate);

    m_Position     = fu->getPositionPointer();
    m_FacingRight  = fu->getFacingRightPointer();

    m_Animator = new Animator(
        texCoords.position.x,
        texCoords.position.y,
        3,                     // frames
        texCoords.size.x * 3,
        texCoords.size.y,
        6);                    // FPS

    // First animation frame
    m_SectionToDraw = m_Animator->getCurrentFrame(false);

    // Allocate **6 vertices instead of 4**
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);

    // Store base UVs (we will overwrite in draw())
    int uPos = texCoords.position.x;
    int vPos = texCoords.position.y;
    int texWidth  = texCoords.size.x;
    int texHeight = texCoords.size.y;

    // Default UVs (facing right, not animated yet)
    float u0 = static_cast<float>(uPos);
    float v0 = static_cast<float>(vPos);
    float u1 = static_cast<float>(uPos + texWidth);
    float v1 = static_cast<float>(vPos + texHeight);

    // ---- TEXCOORDS (two triangles) ----
    // Triangle 1: (0,1,2)
    canvas[m_VertexStartIndex + 0].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 1].texCoords = {u1, v0};
    canvas[m_VertexStartIndex + 2].texCoords = {u1, v1};

    // Triangle 2: (0,2,3)
    canvas[m_VertexStartIndex + 3].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 4].texCoords = {u1, v1};
    canvas[m_VertexStartIndex + 5].texCoords = {u0, v1};
}

void FireballGraphics::draw(VertexArray& canvas)
{
    const Vector2f& pos   = m_Position->position;
    const Vector2f& scale = m_Position->size;

    // ---- POSITIONS (two triangles) ----
    Vector2f p0 = pos;
    Vector2f p1 = pos + Vector2f(scale.x, 0);
    Vector2f p2 = pos + scale;
    Vector2f p3 = pos + Vector2f(0, scale.y);

    // Triangle 1: p0, p1, p2
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // Triangle 2: p0, p2, p3
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;


    // ---- ANIMATION FRAME ----
    m_SectionToDraw = m_Animator->getCurrentFrame(!(*m_FacingRight));

    float uPos = static_cast<float>(m_SectionToDraw->position.x);
    float vPos = static_cast<float>(m_SectionToDraw->position.y);
    float texWidth  = static_cast<float>(m_SectionToDraw->size.x);
    float texHeight = static_cast<float>(m_SectionToDraw->size.y);

    float u0, u1;
    float v0 = vPos;
    float v1 = vPos + texHeight;

    if (*m_FacingRight)
    {
        // Normal orientation
        u0 = uPos;
        u1 = uPos + texWidth;
    }
    else
    {
        // Flip horizontally
        u0 = uPos + texWidth;
        u1 = uPos;
    }

    // ---- UVs (two triangles) ----
    // Triangle 1 (0,1,2)
    canvas[m_VertexStartIndex + 0].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 1].texCoords = {u1, v0};
    canvas[m_VertexStartIndex + 2].texCoords = {u1, v1};

    // Triangle 2 (0,2,3)
    canvas[m_VertexStartIndex + 3].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 4].texCoords = {u1, v1};
    canvas[m_VertexStartIndex + 5].texCoords = {u0, v1};
}
