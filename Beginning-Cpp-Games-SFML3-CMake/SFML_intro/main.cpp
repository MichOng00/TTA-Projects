#include <SFML/Graphics.hpp>
#include <cmath>

int main()
{
    auto window = sf::RenderWindow(sf::VideoMode({800u, 600}), "CMake SFML Project");
    window.setFramerateLimit(144);

    // Create shapes
    sf::CircleShape circle(50);
    circle.setFillColor(sf::Color::Green);
    sf::Vector2f circleStart{100.f, 300.f};
    // circle.setPosition({200, 200});
    circle.setPosition(circleStart);

    sf::RectangleShape rectangle(sf::Vector2f(100, 50));  // Width 100, height 50
    rectangle.setFillColor(sf::Color::Red);
    rectangle.setPosition({400, 300});

    sf::Clock clock; // Track time

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        window.clear();

        // Move the circle with arrow keys
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
            circle.move({0, -5});
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
            circle.move({0, 5});
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
            circle.move({-5, 0});
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
            circle.move({5, 0});

        // Animate the shape (move it in a sinusoidal pattern)
        float time = clock.getElapsedTime().asSeconds();
        rectangle.setPosition({375 + 200 * std::sin(time), 275});

        // Collision detection
        if (circle.getGlobalBounds().findIntersection(rectangle.getGlobalBounds()))
        {
            // Reset circle to starting point
            circle.setPosition(circleStart);
        }

        window.draw(circle);
        window.draw(rectangle);

        window.display();
    }
}