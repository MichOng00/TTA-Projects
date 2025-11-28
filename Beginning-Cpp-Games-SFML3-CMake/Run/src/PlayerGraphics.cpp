#include "PlayerGraphics.h"
#include "PlayerUpdate.h"
#include "Animator.h"
#include "Graphics.h"

void PlayerGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    m_PlayerUpdate = static_pointer_cast<PlayerUpdate>(genericUpdate);
    m_Position = m_PlayerUpdate->getPositionPointer();

    m_Animator = new Animator(
        texCoords.position.x,
        texCoords.position.y,
        6,                       // frames
        texCoords.size.x * 6,    // total width
        texCoords.size.y,
        12);                     // FPS

    m_SectionToDraw = m_Animator->getCurrentFrame(false);
    m_StandingStillSectionToDraw = m_Animator->getCurrentFrame(false);

    // Reserve **6 vertices instead of 4**
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);
}


void PlayerGraphics::draw(VertexArray& canvas)
{
    // ----------- POSITION SETUP (6 vertices) ----------
    const Vector2f& pos  = m_Position->position;
    const Vector2f& size = m_Position->size;

    Vector2f p0 = pos;
    Vector2f p1 = pos + Vector2f(size.x, 0);
    Vector2f p2 = pos + size;
    Vector2f p3 = pos + Vector2f(0, size.y);

    // Triangle 1: 0–1–2
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // Triangle 2: 0–2–3
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;

    // ----------- ANIMATION AND UV SETUP ---------------
    if (m_PlayerUpdate->m_RightIsHeldDown &&
        !m_PlayerUpdate->m_InJump &&
        !m_PlayerUpdate->m_BoostIsHeldDown &&
        m_PlayerUpdate->m_IsGrounded)
    {
        m_SectionToDraw = m_Animator->getCurrentFrame(false);
    }

    if (m_PlayerUpdate->m_LeftIsHeldDown &&
        !m_PlayerUpdate->m_InJump &&
        !m_PlayerUpdate->m_BoostIsHeldDown &&
        m_PlayerUpdate->m_IsGrounded)
    {
        m_SectionToDraw = m_Animator->getCurrentFrame(true);
    }
    else
    {
        m_LastFacingRight = !m_PlayerUpdate->m_LeftIsHeldDown;
    }

    const int uPos = m_SectionToDraw->position.x;
    const int vPos = m_SectionToDraw->position.y;
    const int texWidth = m_SectionToDraw->size.x;
    const int texHeight = m_SectionToDraw->size.y;

    auto setQuadUV = [&](float u0, float v0, float u1, float v1)
    {
        // Triangle 1: 0–1–2
        canvas[m_VertexStartIndex + 0].texCoords = { u0, v0 };
        canvas[m_VertexStartIndex + 1].texCoords = { u1, v0 };
        canvas[m_VertexStartIndex + 2].texCoords = { u1, v1 };

        // Triangle 2: 0–2–3
        canvas[m_VertexStartIndex + 3].texCoords = { u0, v0 };
        canvas[m_VertexStartIndex + 4].texCoords = { u1, v1 };
        canvas[m_VertexStartIndex + 5].texCoords = { u0, v1 };
    };

    // ------------------- RIGHT WALK -------------------
    if (m_PlayerUpdate->m_RightIsHeldDown &&
        !m_PlayerUpdate->m_InJump &&
        !m_PlayerUpdate->m_BoostIsHeldDown)
    {
        setQuadUV(uPos, vPos, uPos + texWidth, vPos + texHeight);
    }

    // ------------------- LEFT WALK -------------------
    else if (m_PlayerUpdate->m_LeftIsHeldDown &&
             !m_PlayerUpdate->m_InJump &&
             !m_PlayerUpdate->m_BoostIsHeldDown)
    {
        setQuadUV(uPos, vPos, uPos - texWidth, vPos + texHeight);
    }

    // ------------------- BOOST RIGHT -------------------
    else if (m_PlayerUpdate->m_RightIsHeldDown &&
             m_PlayerUpdate->m_BoostIsHeldDown)
    {
        setQuadUV(
            BOOST_TEX_LEFT,
            BOOST_TEX_TOP,
            BOOST_TEX_LEFT + BOOST_TEX_WIDTH,
            BOOST_TEX_TOP + BOOST_TEX_HEIGHT
        );
    }

    // ------------------- BOOST LEFT -------------------
    else if (m_PlayerUpdate->m_LeftIsHeldDown &&
             m_PlayerUpdate->m_BoostIsHeldDown)
    {
        setQuadUV(
            BOOST_TEX_LEFT + BOOST_TEX_WIDTH,
            0,
            BOOST_TEX_LEFT,
            100
        );
    }

    // ------------------- VERTICAL BOOST -------------------
    else if (m_PlayerUpdate->m_BoostIsHeldDown)
    {
        setQuadUV(
            536, 0,
            605, 100
        );
    }

    // ------------------- STAND STILL -------------------
    else
    {
        const auto& ss = m_StandingStillSectionToDraw;

        if (m_LastFacingRight)
        {
            setQuadUV(
                ss->position.x,
                ss->position.y,
                ss->position.x + texWidth,
                ss->position.y + texHeight
            );
        }
        else
        {
            setQuadUV(
                ss->position.x + texWidth,
                ss->position.y,
                ss->position.x,
                ss->position.y + texHeight
            );
        }
    }
}
