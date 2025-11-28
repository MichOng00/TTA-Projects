#include "RainGraphics.h"
#include "Animator.h"
#include "Graphics.h"

RainGraphics::RainGraphics(
    FloatRect* playerPosition,
    float horizontalOffset,
    float verticalOffset,
    int rainCoveragePerObject)
{
    m_PlayerPosition = playerPosition;
    m_HorizontalOffset = horizontalOffset;
    m_VerticalOffset = verticalOffset;

    m_Scale.x = rainCoveragePerObject;
    m_Scale.y = rainCoveragePerObject;
}

void RainGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    m_Animator = new Animator(
        texCoords.position.x,
        texCoords.position.y,
        4,                      // frames
        texCoords.size.x * 4,
        texCoords.size.y,
        8);                     // FPS

    // Allocate **6 vertices (triangles)** instead of 4
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);
}

void RainGraphics::draw(VertexArray& canvas)
{
    const Vector2f position =
        m_PlayerPosition->position -
        Vector2f(m_Scale.x / 2 + m_HorizontalOffset,
                 m_Scale.y / 2 + m_VerticalOffset);

    // QUAD CORNERS
    Vector2f p0 = position;
    Vector2f p1 = position + Vector2f(m_Scale.x, 0);
    Vector2f p2 = position + m_Scale;
    Vector2f p3 = position + Vector2f(0, m_Scale.y);

    // -------- POSITION: triangles --------
    // Triangle 1: p0–p1–p2
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // Triangle 2: p0–p2–p3
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;

    // Update frame
    m_SectionToDraw = m_Animator->getCurrentFrame(false);

    const int uPos = m_SectionToDraw->position.x;
    const int vPos = m_SectionToDraw->position.y;
    const int texWidth = m_SectionToDraw->size.x;
    const int texHeight = m_SectionToDraw->size.y;

    float u0 = uPos;
    float v0 = vPos;
    float u1 = uPos + texWidth;
    float v1 = vPos + texHeight;

    // -------- TEXCOORDS: triangles --------
    // Triangle 1 (0–1–2)
    canvas[m_VertexStartIndex + 0].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 1].texCoords = {u1, v0};
    canvas[m_VertexStartIndex + 2].texCoords = {u1, v1};

    // Triangle 2 (0–2–3)
    canvas[m_VertexStartIndex + 3].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 4].texCoords = {u1, v1};
    canvas[m_VertexStartIndex + 5].texCoords = {u0, v1};
}
