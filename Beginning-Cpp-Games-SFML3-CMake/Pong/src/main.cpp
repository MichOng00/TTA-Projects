#include <sstream>
#include <cstdlib>
#include <SFML/Graphics.hpp>
#include "Ball.cpp"
#include "Bat.cpp"
int main()
{
    // Create a video mode object
    VideoMode vm({ 1400, 800 });
    // Create and open a window for the game
    RenderWindow window(vm, "Pong", Style::Default);
    int score = 0;
    int lives = 3;

    // Create a bat at the bottom center of the screen
    Bat bat(1400 / 2, 800 - 20);
    // We will add a ball in the next chapter
    Ball ball(1400 / 2, 0);
    // A cool retro-style font
    Font font("fonts/DS-DIGIT.ttf");
    // Create a Text object called HUD
    Text hud(font);
    // Make it nice and big
    hud.setCharacterSize(75);
    // Choose a color
    hud.setFillColor(Color::White);
    hud.setPosition({20, 20});
    // Here is our clock for timing everything
    Clock clock;
    while (window.isOpen())
    {
        /*
        Handle the player input
        ****************************
        ****************************
        ****************************
        */
        while (const std::optional event = window.pollEvent())
        {
            if (event->is <Event::Closed>())
                // Quit the game when the window is closed
                window.close();
        }
        // Handle the player quitting
        if (Keyboard::isKeyPressed(Keyboard::Key::Enter))
        {
            window.close();
        }
        // Handle the pressing and releasing of the arrow keys
        if (Keyboard::isKeyPressed(Keyboard::Key::Left))
        {
            bat.moveLeft();
        }
        else
        {
            bat.stopLeft();
        }
        if (Keyboard::isKeyPressed(Keyboard::Key::Right))
        {
            bat.moveRight();
        }
        else
        {
            bat.stopRight();
        }


        /*
        Update the bat, the ball and the HUD
        *****************************
        *****************************
        *****************************
        */
        // Update the delta time
        Time dt = clock.restart();
        bat.update(dt);
        ball.update(dt);
        // Update the HUD text
        std::stringstream ss;
        ss << "Score:" << score << "  Lives:" << lives;
        hud.setString(ss.str());

        // Handle ball hitting the bottom
        if (ball.getPosition().position.y > window.getSize().y)
        {
            // reverse the ball direction
            ball.reboundBottom();
            // Remove a life
            lives--;
            // Check for zero lives
            if (lives < 1) {
                // reset the score
                score = 0;
                // reset the lives
                lives = 3;
            }
            // reset ball position
            ball.m_Position = Vector2f({ static_cast<float>(rand() % 1400), 0.0f });
        }

        // Handle ball hitting sides
        if (ball.getPosition().position.x < 0 ||
            ball.getPosition().position.x + ball.getPosition().size.x > window.getSize().x)
        {
            ball.reboundSides();
        }

        // Handle ball hitting top
        if (ball.getPosition().position.y < 0)
        {
            ball.reboundBatOrTop();
        }

        // Has the ball hit the bat?
        if (ball.getPosition().findIntersection(bat.getPosition()))
        {
            // Hit detected so reverse the ball and score a point
            ball.reboundBatOrTop();
			score++;
        }

        // Limit the bat to the window
        if (bat.getPosition().position.x < 0)
        {
            bat.m_Position = Vector2f({ 0, bat.getPosition().position.y });
        }
        else if (bat.getPosition().position.x > 1400 - bat.getPosition().size.x)
        {
            bat.m_Position = Vector2f({ 1400 - bat.getPosition().size.x, bat.getPosition().position.y });
        }
        


        /*
        Draw the bat, the ball and the HUD
        *****************************
        *****************************
        *****************************
        */
        window.clear();
        window.draw(hud);
        window.draw(bat.getShape());
        window.draw(ball.getShape());
        window.display();



    }
    return 0;
}
