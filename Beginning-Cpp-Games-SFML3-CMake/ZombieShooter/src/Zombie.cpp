#include "Zombie.h"
#include "TextureHolder.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

// -------------------------------------------
// Default constructor (required for new Zombie[num])
// -------------------------------------------
Zombie::Zombie()
    : m_Sprite(TextureHolder::GetTexture("graphics/bloater.png"))
{
    m_Position = {0.f, 0.f};
    m_Alive = false;

    m_Sprite = sf::Sprite(TextureHolder::GetTexture("graphics/bloater.png"));
    m_Sprite.setOrigin({25.f, 25.f});

    m_Speed = 0.f;
    m_Health = 0;
}

// -------------------------------------------
// Spawn a zombie of a specific type
// -------------------------------------------
void Zombie::spawn(float startX, float startY, int type, int seed)
{
    switch (type)
    {
    case 0:
        // Bloater
        m_Sprite = sf::Sprite(TextureHolder::GetTexture("graphics/bloater.png"));
        m_Speed = BLOATER_SPEED;
        m_Health = BLOATER_HEALTH;
        break;

    case 1:
        // Chaser
        m_Sprite = sf::Sprite(TextureHolder::GetTexture("graphics/chaser.png"));
        m_Speed = CHASER_SPEED;
        m_Health = CHASER_HEALTH;
        break;

    case 2:
        // Crawler
        m_Sprite = sf::Sprite(TextureHolder::GetTexture("graphics/crawler.png"));
        m_Speed = CRAWLER_SPEED;
        m_Health = CRAWLER_HEALTH;
        break;
    }

    // Unique speed modifier
    srand(static_cast<unsigned>(time(0)) * seed);
    float modifier = (rand() % MAX_VARRIANCE) + OFFSET;   // 80–100
    modifier /= 100.f;
    m_Speed *= modifier;

    // Position
    m_Position = {startX, startY};

    m_Sprite.setOrigin({25.f, 25.f});
    m_Sprite.setPosition(m_Position);

    m_Alive = true;
}

// -------------------------------------------
// Take damage
// -------------------------------------------
bool Zombie::hit()
{
    m_Health--;

    if (m_Health < 0)
    {
        m_Alive = false;
        m_Sprite.setTexture(TextureHolder::GetTexture("graphics/blood.png"));
        return true;    // zombie died
    }

    return false;       // injured but alive
}

// -------------------------------------------
// Queries
// -------------------------------------------
bool Zombie::isAlive()
{
    return m_Alive;
}

sf::FloatRect Zombie::getPosition()
{
    return m_Sprite.getGlobalBounds();
}

sf::Sprite Zombie::getSprite()
{
    return m_Sprite;
}

// -------------------------------------------
// Update movement + direction
// -------------------------------------------
void Zombie::update(float elapsedTime, sf::Vector2f playerLocation)
{
    float playerX = playerLocation.x;
    float playerY = playerLocation.y;

    // Move toward the player
    if (playerX > m_Position.x)  m_Position.x += m_Speed * elapsedTime;
    if (playerX < m_Position.x)  m_Position.x -= m_Speed * elapsedTime;
    if (playerY > m_Position.y)  m_Position.y += m_Speed * elapsedTime;
    if (playerY < m_Position.y)  m_Position.y -= m_Speed * elapsedTime;

    // Apply updated position
    m_Sprite.setPosition(m_Position);

    // Face the player (SFML-3 requires sf::degrees)
    float angle = std::atan2(playerY - m_Position.y,
                             playerX - m_Position.x) * 180.f / 3.14159265f;

    m_Sprite.setRotation(sf::degrees(angle));
}
