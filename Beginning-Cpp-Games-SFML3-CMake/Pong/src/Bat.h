#pragma once // prevent compiler processing >= 1 time
#include <SFML/Graphics.hpp>
using namespace sf;
class Bat
{
private:
    // A RectangleShape object
    RectangleShape m_Shape;
    float m_Speed = 1000.0f;
    bool m_MovingRight = false;
    bool m_MovingLeft = false;
public:
    Vector2f m_Position;
    Bat(float startX, float startY);
    FloatRect getPosition();
    RectangleShape getShape();
    void moveLeft();
    void moveRight();
    void stopLeft();
    void stopRight();
    void update(Time dt);
};
