#include "PlayerUpdate.h"
#include "SoundEngine.h"
#include "LevelUpdate.h"
#include "InputReceiver.cpp"

FloatRect* PlayerUpdate::getPositionPointer()
{
    return &m_Position;
}

bool* PlayerUpdate::getGroundedPointer()
{
    return &m_IsGrounded;
}

InputReceiver* PlayerUpdate::getInputReceiver()
{
    return &m_InputReceiver;

}

void PlayerUpdate::assemble(
    shared_ptr<LevelUpdate> levelUpdate,
    shared_ptr<PlayerUpdate> playerUpdate)
{
    // SoundEngine::SoundEngine();

    m_Position.size.x = PLAYER_WIDTH;
    m_Position.size.y = PLAYER_HEIGHT;
    m_IsPaused = levelUpdate->getIsPausedPointer();
}

void PlayerUpdate::handleInput()
{
    for (const Event& event : m_InputReceiver.getEvents())
    {
        if (event.is<sf::Event::KeyPressed>())
        {
            const auto key = event.getIf<sf::Event::KeyPressed>();
            if (!key) continue;

            if (key->code == Keyboard::Key::D)
            {
                m_RightIsHeldDown = true;
            }
            if (key->code == Keyboard::Key::A)
            {
                m_LeftIsHeldDown = true;
            }

            if (key->code == Keyboard::Key::W)
            {
                m_BoostIsHeldDown = true;
            }

            if (key->code == Keyboard::Key::Space)
            {
                m_SpaceHeldDown = true;
            }
        }

        if (event.is<sf::Event::KeyReleased>())
        {
            const auto key = event.getIf<sf::Event::KeyReleased>();
            if (!key) continue;
            if (key->code == Keyboard::Key::D)
            {
                m_RightIsHeldDown = false;
            }
            if (key->code == Keyboard::Key::A)
            {
                m_LeftIsHeldDown = false;
            }

            if (key->code == Keyboard::Key::W)
            {
                m_BoostIsHeldDown = false;
            }

            if (key->code == Keyboard::Key::Space)
            {
                m_SpaceHeldDown = false;
            }

        }
    }

    m_InputReceiver.clearEvents();
}


void PlayerUpdate::update(float timeTakenThisFrame)
{
    if (!*m_IsPaused)
    {
        // All the rest of the code is in here
        m_Position.position.y += m_Gravity *
            timeTakenThisFrame;

        handleInput();

        if (m_IsGrounded)
        {
            if (m_RightIsHeldDown)
            {
                m_Position.position.x +=
                    timeTakenThisFrame * m_RunSpeed;
            }

            if (m_LeftIsHeldDown)
            {
                m_Position.position.x -=
                    timeTakenThisFrame * m_RunSpeed;
            }
        }

        if (m_BoostIsHeldDown)
        {
            m_Position.position.y -=
                timeTakenThisFrame * m_BoostSpeed;

            if (m_RightIsHeldDown)
            {
                m_Position.position.x +=
                    timeTakenThisFrame * m_RunSpeed / 2;
            }

            if (m_LeftIsHeldDown)
            {
                m_Position.position.x -=
                    timeTakenThisFrame * m_RunSpeed / 4;
            }
        }

        // Handle Jumping
        if (m_SpaceHeldDown && !m_InJump && m_IsGrounded)
        {
            SoundEngine::playJump();
            m_InJump = true;
            m_JumpClock.restart();
        }

        if (!m_SpaceHeldDown)
        {
            //mInJump = false;
        }

        if (m_InJump)
        {
            if (m_JumpClock.getElapsedTime().asSeconds() <
                m_JumpDuration / 2)
            {
                // Going up
                m_Position.position.y -= m_JumpSpeed *
                    timeTakenThisFrame;
            }
            else
            {
                // Going down
                m_Position.position.y +=
                    m_JumpSpeed * timeTakenThisFrame;
            }

            if (m_JumpClock.getElapsedTime().asSeconds() >
                m_JumpDuration)
            {
                m_InJump = false;
            }

            if (m_RightIsHeldDown)
            {
                m_Position.position.x +=
                    timeTakenThisFrame * m_RunSpeed;
            }

            if (m_LeftIsHeldDown)
            {
                m_Position.position.x -=
                    timeTakenThisFrame * m_RunSpeed;
            }
        }// End if (m_InJump)

        m_IsGrounded = false;
    }
}



