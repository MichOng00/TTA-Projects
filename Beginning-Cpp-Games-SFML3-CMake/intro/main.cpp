#include <SFML/Graphics.hpp>
#include <cmath>

int main() {
	auto window = sf::RenderWindow(sf::VideoMode({ 800, 600 }), "SFML Window");
	window.setFramerateLimit(144);

	// Circle
	sf::CircleShape circle(50);
	circle.setFillColor(sf::Color::Green);
	circle.setPosition({ 200, 200 });

	// Rectangle
	sf::RectangleShape rectangle({ 100, 50 });
	rectangle.setFillColor(sf::Color::Red);
	rectangle.setPosition({ 400, 300 });

	sf::Clock clock;

	while (window.isOpen()) {
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) {
				window.close();
			}
			// Circle moves to where the mouse is clicked
			if (auto* mouse = event->getIf<sf::Event::MouseButtonPressed>()) {
				if (mouse->button == sf::Mouse::Button::Left) {
					sf::Vector2f pos{ (float)mouse->position.x, (float)mouse->position.y };
					circle.setPosition(pos);
				}
			}
		}
		window.clear();

		// Move the circle
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
			circle.move({ 0, -5 });
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
			circle.move({ 0, 5 });
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
			circle.move({ -5, 0 });
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
			circle.move({ 5, 0 });

		// Move the rectangle
		float time = clock.getElapsedTime().asSeconds();
		rectangle.setPosition({ 200 + 50 * std::sin(time), 200 });

		// Collision detection
		if (circle.getGlobalBounds().findIntersection(rectangle.getGlobalBounds())) {
			circle.setFillColor(sf::Color::Yellow);
		} else {
			circle.setFillColor(sf::Color::Green);
		}

		// Draw everything
		window.draw(circle);
		window.draw(rectangle);

		window.display();
	}
}