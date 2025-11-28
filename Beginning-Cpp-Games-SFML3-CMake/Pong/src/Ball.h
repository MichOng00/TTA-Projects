#pragma once
#include <SFML/Graphics.hpp>
using namespace sf;
class Ball
{
private:
    RectangleShape m_Shape;
    float m_Speed = 400.0f;
    float m_DirectionX = .2f;
    float m_DirectionY = .2f;
public:
    Ball(float startX, float startY);
    Vector2f m_Position;
    FloatRect getPosition();
    RectangleShape getShape();
    float getXVelocity();
    void reboundSides();
    void reboundBatOrTop();
    void reboundBottom();
    void update(Time dt);
};
