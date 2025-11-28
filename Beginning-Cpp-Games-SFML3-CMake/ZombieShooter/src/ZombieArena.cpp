#include <sstream>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include "TextureHolder.cpp"
#include "ZombieArena.h"
#include "Player.cpp"
#include "Zombie.cpp"
#include "CreateHorde.cpp"
#include "CreateBackground.cpp"
#include "Bullet.cpp"
#include "Pickup.cpp"
#include <iostream>

using namespace sf;

int main()
{
    // Instance of TextureHolder (unchanged)
    TextureHolder holder;

    // Game states
    enum class State {
        PAUSED, LEVELING_UP,
        GAME_OVER, PLAYING
    };

    State state = State::GAME_OVER;

    // Get screen resolution and create window
    sf::Vector2u resolution = sf::VideoMode::getDesktopMode().size;

    sf::RenderWindow window(
        sf::VideoMode(resolution),
        "Zombie Arena",
        sf::Style::Default
    );

    // Main view (use FloatRect)
    sf::View mainView(sf::FloatRect(
        { 0.f, 0.f},
        {static_cast<float>(resolution.x), static_cast<float>(resolution.y)}
    ));


    // Clock & timing
    Clock clock;
    Time gameTimeTotal;

    // Mouse positions
    Vector2f mouseWorldPosition;
    Vector2i mouseScreenPosition;

    // Player
    Player player;

    // Arena: note sf::Rect refactor in SFML 3 -> use position and size members.
    IntRect arena; // still the same type name; access its .position/.size members below.

    // Background
    VertexArray background;
    Texture textureBackground("graphics/background_sheet.png");

    // Zombies
    int numZombies = 0;
    int numZombiesAlive = 0;
    Zombie* zombies = nullptr;

    // Bullets
    Bullet bullets[100];
    int currentBullet = 0;
    int bulletsSpare = 24;
    int bulletsInClip = 6;
    int clipSize = 6;
    float fireRate = 1.f;
    Time lastPressed;

    // Crosshair & mouse cursor
    window.setMouseCursorVisible(true);
    Texture textureCrosshair("graphics/crosshair.png");
    Sprite spriteCrosshair(textureCrosshair);
    spriteCrosshair.setOrigin({25.f, 25.f});

    // Pickups
    Pickup healthPickup(1);
    Pickup ammoPickup(2);

    // Score & hi-score
    int score = 0;
    int hiScore = 0;

    // Game-over / HUD assets
    Texture textureGameOver("graphics/background.png");
    Sprite spriteGameOver(textureGameOver);
    spriteGameOver.setPosition({0.f, 0.f});

    View hudView(FloatRect({0.f, 0.f}, {1920.f, 1080.f}));

    Texture textureAmmoIcon("graphics/ammo_icon.png");
    Sprite spriteAmmoIcon(textureAmmoIcon);
    spriteAmmoIcon.setPosition({20.f, 980.f});

    Font font("fonts/zombiecontrol.ttf");

    // Texts
    Text pausedText(font);
    pausedText.setCharacterSize(155);
    pausedText.setFillColor(Color::White);
    pausedText.setPosition({400.f, 400.f});
    pausedText.setString("Press Enter \nto continue");

    Text gameOverText(font);
    gameOverText.setCharacterSize(125);
    gameOverText.setFillColor(Color::White);
    gameOverText.setPosition({250.f, 850.f});
    gameOverText.setString("Press Enter to play");

    Text levelUpText(font);
    levelUpText.setCharacterSize(80);
    levelUpText.setFillColor(Color::White);
    levelUpText.setPosition({150.f, 250.f});
    std::stringstream levelUpStream;
    levelUpStream <<
        "1- Increased rate of fire" <<
        "\n2- Increased clip size(next reload)" <<
        "\n3- Increased max health" <<
        "\n4- Increased run speed" <<
        "\n5- More and better health pickups" <<
        "\n6- More and better ammo pickups";
    levelUpText.setString(levelUpStream.str());

    Text ammoText(font);
    ammoText.setCharacterSize(55);
    ammoText.setFillColor(Color::White);
    ammoText.setPosition({200.f, 980.f});

    Text scoreText(font);
    scoreText.setCharacterSize(55);
    scoreText.setFillColor(Color::White);
    scoreText.setPosition({20.f, 0.f});

    //-----------------------------------------------------
    // HUD: Controls / Instructions
    //-----------------------------------------------------
    Text controlsText(font);
    controlsText.setCharacterSize(40);
    controlsText.setFillColor(Color::White);
    controlsText.setPosition({20.f, 900.f});
    controlsText.setString(
        "WASD: Move     Mouse: Aim     LMB: Shoot\n"
        "R: Reload     ENTER: Pause"
    );


    // Load hi score
    std::ifstream inputFile("gamedata/scores.txt");
    if (inputFile.is_open())
    {
        inputFile >> hiScore;
        inputFile.close();
    }

    Text hiScoreText(font);
    hiScoreText.setCharacterSize(55);
    hiScoreText.setFillColor(Color::White);
    hiScoreText.setPosition({1400.f, 0.f});
    {
        std::stringstream s;
        s << "Hi Score:" << hiScore;
        hiScoreText.setString(s.str());
    }

    Text zombiesRemainingText(font);
    zombiesRemainingText.setCharacterSize(55);
    zombiesRemainingText.setFillColor(Color::White);
    zombiesRemainingText.setPosition({1500.f, 980.f});
    zombiesRemainingText.setString("Zombies: 100");

    int wave = 0;
    Text waveNumberText(font);
    waveNumberText.setCharacterSize(55);
    waveNumberText.setFillColor(Color::White);
    waveNumberText.setPosition({1250.f, 980.f});
    waveNumberText.setString("Wave: 0");

    RectangleShape healthBar;
    healthBar.setFillColor(Color::Red);
    healthBar.setPosition({450.f, 980.f});

    int framesSinceLastHUDUpdate = 0;
    int fpsMeasurementFrameInterval = 1000;

    // Sounds
    SoundBuffer hitBuffer("sound/hit.wav");
    Sound hit(hitBuffer);

    SoundBuffer splatBuffer("sound/splat.wav");
    Sound splat(splatBuffer);

    SoundBuffer shootBuffer("sound/shoot.wav");
    Sound shoot(shootBuffer);

    SoundBuffer reloadBuffer("sound/reload.wav");
    Sound reload(reloadBuffer);

    SoundBuffer reloadFailedBuffer("sound/reload_failed.wav");
    Sound reloadFailed(reloadFailedBuffer);

    SoundBuffer powerupBuffer("sound/powerup.wav");
    Sound powerup(powerupBuffer);

    SoundBuffer pickupBuffer("sound/pickup.wav");
    Sound pickup(pickupBuffer);

    // Main game loop
    while (window.isOpen())
    {
        /************
         Handle input
        ************/

        // SFML 3: pollEvent returns std::optional<sf::Event>.
        while (const std::optional event = window.pollEvent())
        {
            // Window closed?
            if (event->is<sf::Event::Closed>())
            {
                window.close();
                continue;
            }

            // Key pressed? use is/getIf pattern. (SFML 3: std::variant-backed events)
            if (event->is<sf::Event::KeyPressed>())
            {
                const auto key = event->getIf<sf::Event::KeyPressed>();
                if (!key) continue;

                // ENTER handling (use scoped enum)
                if (key->code == sf::Keyboard::Key::Enter &&
                    state == State::PLAYING)
                {
                    state = State::PAUSED;
                }
                else if (key->code == sf::Keyboard::Key::Enter &&
                         state == State::PAUSED)
                {
                    state = State::PLAYING;
                    // Reset the clock so there isn't a frame jump
                    clock.restart();
                }
                else if (key->code == sf::Keyboard::Key::Enter &&
                         state == State::GAME_OVER)
                {
                    state = State::LEVELING_UP;
                    wave = 0;
                    score = 0;
                    currentBullet = 0;
                    bulletsSpare = 24;
                    bulletsInClip = 6;
                    clipSize = 6;
                    fireRate = 1.f;
                    player.resetPlayerStats();
                }

                // If playing, handle reload key (R)
                if (state == State::PLAYING &&
                    key->code == sf::Keyboard::Key::R)
                {
                    if (bulletsSpare >= clipSize)
                    {
                        bulletsInClip = clipSize;
                        bulletsSpare -= clipSize;
                        reload.play();
                    }
                    else if (bulletsSpare > 0)
                    {
                        bulletsInClip = bulletsSpare;
                        bulletsSpare = 0;
                        reload.play();
                    }
                    else
                    {
                        reloadFailed.play();
                    }
                }

                // LEVELING_UP numeric choices must be handled here (moved from outside)
                if (state == State::LEVELING_UP)
                {
                    if (key->code == sf::Keyboard::Key::Num1)
                    {
                        fireRate++;
                        state = State::PLAYING;
                    }
                    else if (key->code == sf::Keyboard::Key::Num2)
                    {
                        clipSize += clipSize;
                        state = State::PLAYING;
                    }
                    else if (key->code == sf::Keyboard::Key::Num3)
                    {
                        player.upgradeHealth();
                        state = State::PLAYING;
                    }
                    else if (key->code == sf::Keyboard::Key::Num4)
                    {
                        player.upgradeSpeed();
                        state = State::PLAYING;
                    }
                    else if (key->code == sf::Keyboard::Key::Num5)
                    {
                        healthPickup.upgrade();
                        state = State::PLAYING;
                    }
                    else if (key->code == sf::Keyboard::Key::Num6)
                    {
                        ammoPickup.upgrade();
                        state = State::PLAYING;
                    }

                    // If we just moved to PLAYING, prepare the level
                    if (state == State::PLAYING)
                    {
                        // Increase the wave number
                        wave++;

                        // Adapted to new sf::Rect fields: position & size.
                        arena.size.x = 500 * wave;
                        arena.size.y = 500 * wave;
                        arena.position.x = 0;
                        arena.position.y = 0;

                        int tileSize = createBackground(background, arena);

                        player.spawn(arena, resolution, tileSize);

                        healthPickup.setArena(arena);
                        ammoPickup.setArena(arena);

                        numZombies = 5 * wave;

                        delete[] zombies;
                        zombies = createHorde(numZombies, arena);
                        numZombiesAlive = numZombies;

                        powerup.play();

                        clock.restart();
                    }
                } // end LEVELING_UP handling
            } // end key pressed
        } // end event polling

        /****************
         Game input while PLAYING (continuous)
        ****************/
        if (state == State::PLAYING)
        {
            // WASD continuous movement
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) player.moveUp();
            else player.stopUp();

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) player.moveDown();
            else player.stopDown();

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) player.moveLeft();
            else player.stopLeft();

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) player.moveRight();
            else player.stopRight();

            // Fire a bullet: use mouse position relative to the window and map to world coords
            // Use sf::Mouse::getPosition(window) so it's window-relative pixel coords before mapping
            mouseScreenPosition = sf::Mouse::getPosition(window);
            mouseWorldPosition = window.mapPixelToCoords(mouseScreenPosition, mainView);
            spriteCrosshair.setPosition(mouseWorldPosition);

            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
            {
                if (gameTimeTotal.asMilliseconds() - lastPressed.asMilliseconds()
                    > 1000.f / fireRate && bulletsInClip > 0)
                {
                    bullets[currentBullet].shoot(
                        player.getCenter().x, player.getCenter().y,
                        mouseWorldPosition.x, mouseWorldPosition.y);
                    currentBullet++;
                    if (currentBullet > 99) currentBullet = 0;
                    lastPressed = gameTimeTotal;
                    shoot.play();
                    bulletsInClip--;
                }
            }
        } // end PLAYING input

        /*
         UPDATE THE FRAME
        */
        if (state == State::PLAYING)
        {
            // Update delta time
            Time dt = clock.restart();
            gameTimeTotal += dt;
            float dtAsSeconds = dt.asSeconds();

            // If the crosshair position hasn't been set above (e.g. in other states), update here too:
            mouseScreenPosition = sf::Mouse::getPosition(window);
            mouseWorldPosition = window.mapPixelToCoords(mouseScreenPosition, mainView);
            spriteCrosshair.setPosition(mouseWorldPosition);

            // Update player, zombies, bullets, pickups
            player.update(dtAsSeconds, sf::Mouse::getPosition(window));
            Vector2f playerPosition(player.getCenter());
            mainView.setCenter(player.getCenter());

            for (int i = 0; i < numZombies; ++i)
            {
                if (zombies && zombies[i].isAlive())
                {
                    zombies[i].update(dtAsSeconds, playerPosition);
                }
            }

            for (int i = 0; i < 100; ++i)
            {
                if (bullets[i].isInFlight())
                    bullets[i].update(dtAsSeconds);
            }

            healthPickup.update(dtAsSeconds);
            ammoPickup.update(dtAsSeconds);

            // Bullet-vs-zombie collision
            for (int i = 0; i < 100; ++i)
            {
                for (int j = 0; j < numZombies; ++j)
                {
                    if (bullets[i].isInFlight() && zombies[j].isAlive())
                    {
                        if (bullets[i].getPosition().findIntersection(zombies[j].getPosition()))
                        {
                            bullets[i].stop();
                            if (zombies[j].hit())
                            {
                                score += 10;
                                if (score >= hiScore) hiScore = score;
                                numZombiesAlive--;
                                if (numZombiesAlive == 0) state = State::LEVELING_UP;
                                splat.play();
                            }
                        }
                    }
                }
            }

            // Zombies touching player
            for (int i = 0; i < numZombies; ++i)
            {
                if (zombies && player.getPosition().findIntersection(zombies[i].getPosition()) && zombies[i].isAlive())
                {
                    if (player.hit(gameTimeTotal))
                    {
                        hit.play();
                    }
                    if (player.getHealth() <= 0)
                    {
                        state = State::GAME_OVER;
                        std::ofstream outputFile("gamedata/scores.txt");
                        outputFile << hiScore;
                        outputFile.close();
                    }
                }
            }

            // Health pickup
            if (player.getPosition().findIntersection(healthPickup.getPosition()) && healthPickup.isSpawned())
            {
                player.increaseHealthLevel(healthPickup.gotIt());
                pickup.play();
            }

            // Ammo pickup
            if (player.getPosition().findIntersection(ammoPickup.getPosition()) && ammoPickup.isSpawned())
            {
                bulletsSpare += ammoPickup.gotIt();
                reload.play();
            }

            // Update HUD values occasionally
            healthBar.setSize({player.getHealth() * 3.f, 50.f});
            framesSinceLastHUDUpdate++;
            if (framesSinceLastHUDUpdate > fpsMeasurementFrameInterval)
            {
                std::stringstream ssAmmo, ssScore, ssHiScore, ssWave, ssZombiesAlive;
                ssAmmo << bulletsInClip << "/" << bulletsSpare;
                ammoText.setString(ssAmmo.str());
                ssScore << "Score:" << score;
                scoreText.setString(ssScore.str());
                ssHiScore << "Hi Score:" << hiScore;
                hiScoreText.setString(ssHiScore.str());
                ssWave << "Wave:" << wave;
                waveNumberText.setString(ssWave.str());
                ssZombiesAlive << "Zombies:" << numZombiesAlive;
                zombiesRemainingText.setString(ssZombiesAlive.str());
                framesSinceLastHUDUpdate = 0;
            }
        } // end update

        /*
         DRAW THE SCENE
        */
        window.clear();

        if (state == State::PLAYING)
        {
            window.setView(mainView);
            window.draw(background, &textureBackground);

            for (int i = 0; i < numZombies; ++i)
                if (zombies) window.draw(zombies[i].getSprite());

            for (int i = 0; i < 100; ++i)
                if (bullets[i].isInFlight()) window.draw(bullets[i].getShape());

            if (ammoPickup.isSpawned()) window.draw(ammoPickup.getSprite());
            if (healthPickup.isSpawned()) window.draw(healthPickup.getSprite());

            window.draw(spriteCrosshair);
            window.draw(player.getSprite());

            window.setView(hudView);
            window.draw(spriteAmmoIcon);
            window.draw(ammoText);
            window.draw(scoreText);
            window.draw(hiScoreText);
            window.draw(healthBar);
            window.draw(waveNumberText);
            window.draw(zombiesRemainingText);
            window.draw(controlsText);

        }
        else if (state == State::LEVELING_UP)
        {
            window.setView(hudView);
            window.draw(spriteGameOver);
            window.draw(levelUpText);
        }
        else if (state == State::PAUSED)
        {
            window.setView(hudView);
            window.draw(pausedText);
        }
        else if (state == State::GAME_OVER)
        {
            window.setView(hudView);
            window.draw(spriteGameOver);
            window.draw(gameOverText);
            window.draw(scoreText);
            window.draw(hiScoreText);
        }

        window.display();
    } // end main loop

    delete[] zombies;
    return 0;
}
