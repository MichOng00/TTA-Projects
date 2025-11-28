#include "CameraUpdate.h"
#include "PlayerUpdate.h"

FloatRect* CameraUpdate::getPositionPointer()
{
    return &m_Position;
}

void CameraUpdate::assemble(
    shared_ptr<LevelUpdate> levelUpdate,
    shared_ptr<PlayerUpdate> playerUpdate)
{
    m_PlayerPosition =
        playerUpdate->getPositionPointer();
}

InputReceiver* CameraUpdate::getInputReceiver()
{
    m_InputReceiver = new InputReceiver;
    m_ReceivesInput = true;
    return m_InputReceiver;
}

void CameraUpdate::handleInput()
{
    m_Position.size.x = 1.0f;

    // FIX: m_InputReceiver is a pointer → use ->
    for (const sf::Event& event : m_InputReceiver->getEvents())
    {
        // Correct SFML 3 wheel-scrolled event
        if (event.is<sf::Event::MouseWheelScrolled>())
        {
            const auto* wheel = event.getIf<sf::Event::MouseWheelScrolled>();
            if (!wheel) continue;

            if (wheel->wheel == sf::Mouse::Wheel::Vertical)
            {
                // Zoom in/out
                m_Position.size.x *= (wheel->delta > 0) ? 0.95f : 1.05f;
            }
        }
    }

    // FIX: Clear events AFTER processing
    m_InputReceiver->clearEvents();
}


void CameraUpdate::update(float fps)
{
    if (m_ReceivesInput)
    {
        handleInput();

        m_Position.position.x = m_PlayerPosition->position.x;
        m_Position.position.y = m_PlayerPosition->position.y;
    }
    else
    {
        m_Position.position.x = m_PlayerPosition->position.x;
        m_Position.position.y = m_PlayerPosition->position.y;
        m_Position.size.x = 1;
    }
}
