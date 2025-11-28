#include <sstream>
#include <random>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

// Make code easier to type with "using namespace"
using namespace sf;

// Function declaration
void updateBranches(int seed);

const int NUM_BRANCHES = 6;
std::vector<Sprite> branches;

// Where is the player/branch? 
// Left or Right
enum class side { LEFT, RIGHT, NONE };
side branchPositions[NUM_BRANCHES];

int main()
{
    // Create a video mode object
    VideoMode vm({1600, 900});

    // Create and open a window for the game
    RenderWindow window(vm, "Timber!!!", Style::Default);
    window.setFramerateLimit(60);

    // Create a texture to hold a graphic on the GPU 
    Texture textureBackground("graphics/background.png");

    // Create a sprite and attach the texture to the sprite 
    Sprite spriteBackground(textureBackground);
    spriteBackground.setPosition({0.f, 0.f});

    // Make a tree sprite
    Texture textureTree("graphics/tree.png");
    Sprite spriteTree(textureTree);
    spriteTree.setPosition({700.f, 0.f});

    // Enhanced version (additional background trees)
    Texture textureTree2("graphics/tree2.png");
    Sprite spriteTree2(textureTree2);
    Sprite spriteTree3(textureTree2);
    Sprite spriteTree4(textureTree2);
    Sprite spriteTree5(textureTree2);
    Sprite spriteTree6(textureTree2);

    spriteTree2.setPosition({20.f, 0.f});
    spriteTree3.setPosition({350.f, -400.f});
    spriteTree4.setPosition({1000.f, -400.f});
    spriteTree5.setPosition({1300.f, -500.f});
    spriteTree6.setPosition({1600.f, 0.f});

    // Prepare the bee
    Texture textureBee("graphics/bee.png");
    Sprite spriteBee(textureBee);
    spriteBee.setPosition({ 0.f, 700.f });

    // Is the bee currently moving?
    bool beeActive = false;
    // How fast can the bee fly
    float beeSpeed = 0.0f;

    // Cloud texture + arrays
    Texture textureCloud("graphics/cloud.png");

    const int NUM_CLOUDS = 6;
    std::vector<Sprite> clouds;
    clouds.reserve(NUM_CLOUDS);
    for (int i = 0; i < NUM_CLOUDS; ++i)
    {
        clouds.emplace_back(textureCloud);
        // Position the cloud on the left of the screen at different heights
        clouds[i].setPosition({-300.f, static_cast<float>(i * 120)});
    }
    // Are the clouds currently on screen?
    std::vector<bool> cloudsActive(NUM_CLOUDS, false);
    // How fast is each cloud?
    std::vector<float> cloudSpeeds(NUM_CLOUDS, 0.0f);

    // Variables to control time itself
    Clock clock;

    // Time bar
    RectangleShape timeBar;
    float timeBarStartWidth = 400.f;
    float timeBarHeight = 80.f;
    timeBar.setSize({timeBarStartWidth, timeBarHeight});
    timeBar.setFillColor(Color::Red);
    timeBar.setPosition({(1600.f / 2.f) - timeBarStartWidth / 2.f, 850.f});

    float timeRemaining = 6.0f;
    float timeBarWidthPerSecond = timeBarStartWidth / timeRemaining;

    // Track whether the game is running
    bool paused = true;

    // Score
    int score = 0;

    // We need to choose a font
    Font font("fonts/KOMIKAP_.ttf");

    Text messageText(font);
    Text scoreText(font);
    Text fpsText(font);

    // Set up the fps text
    fpsText.setFillColor(Color::White);
    fpsText.setCharacterSize(50);
    fpsText.setPosition({1300.f, 20.f});

    // Assign the actual message
    messageText.setString("Press Enter to start!");
    scoreText.setString("Score = 0");

    // Character sizes
    messageText.setCharacterSize(60);
    scoreText.setCharacterSize(80);

    // Colors
    messageText.setFillColor(Color::White);
    scoreText.setFillColor(Color::White);

    // Center the message text
    FloatRect textRect = messageText.getLocalBounds();
    messageText.setOrigin({textRect.position.x + textRect.size.x / 2.0f, textRect.position.y + textRect.size.y / 2.0f});
    messageText.setPosition({1600.f / 2.0f, 900.f / 2.0f});

    scoreText.setPosition({20.f, 20.f});

    // Backgrounds for the text
    RectangleShape rect1;
    rect1.setFillColor(Color(0, 0, 0, 150));
    rect1.setSize({500.f, 90.f});
    rect1.setPosition({0.f, 30.f});

    RectangleShape rect2;
    rect2.setFillColor(Color(0, 0, 0, 150));
    rect2.setSize({8000.f, 90.f});
    rect2.setPosition({1150.f, 30.f});

    // Prepare branches
    Texture textureBranch("graphics/branch.png");

    branches.reserve(NUM_BRANCHES);
    for (int i = 0; i < NUM_BRANCHES; i++)
    {
        branches.emplace_back(textureBranch);
        branches[i].setPosition({ -2000.f, -2000.f });
        branches[i].setOrigin({ 200.f, 20.f });
        branchPositions[i] = side::NONE;
    }


    // Prepare the player
    Texture texturePlayer("graphics/player.png");
    Sprite spritePlayer(texturePlayer);
    spritePlayer.setPosition({480.f, 600.f});

    // The player starts on the left
    side playerSide = side::LEFT;

    // Prepare the gravestone
    Texture textureRIP("graphics/rip.png");
    Sprite spriteRIP(textureRIP);
    spriteRIP.setPosition({500.f, 700.f});

    // Prepare the axe
    Texture textureAxe("graphics/axe.png");
    Sprite spriteAxe(textureAxe);
    spriteAxe.setPosition({600.f, 700.f});

    // Line the axe up with the tree
    const float AXE_POSITION_LEFT = 580.f;
    const float AXE_POSITION_RIGHT = 890.f;

    // Prepare the flying log
    Texture textureLog("graphics/log.png");
    Sprite spriteLog(textureLog);
    spriteLog.setPosition({700.f, 600.f});

    // Some other useful log related variables
    bool logActive = false;
    float logSpeedX = 1000.f;
    float logSpeedY = -1500.f;

    // Control the player input
    bool acceptInput = false;

    // Prepare the sound
    SoundBuffer chopBuffer("sound/chop.wav");
    Sound chop(chopBuffer);

    SoundBuffer deathBuffer("sound/death.wav");
    Sound death(deathBuffer);

    // Out of time
    SoundBuffer ootBuffer("sound/out_of_time.wav");
    Sound outOfTime(ootBuffer);

    // control the drawing of the score
    int lastDrawn = 0;

    // Random number generator (better than repeated srand)
    std::random_device rd;
    std::mt19937 rng(rd());

    while (window.isOpen())
    {
        //Event event;
        while (const std::optional event = window.pollEvent())
        {
            // Window closed
            if (event->is<sf::Event::Closed>())
                window.close();

            // Key released event used to re-enable input
            if (event->is<sf::Event::KeyReleased>() && !paused)
            {
                acceptInput = true;

                // hide the axe off-screen
                spriteAxe.setPosition({2000.f, spriteAxe.getPosition().y});
            }

            // Escape to close
            if (event->is<sf::Event::KeyPressed>())
            {
                if (const auto* keyEvent = event->getIf<sf::Event::KeyPressed>())
                {
                    if (keyEvent->scancode == sf::Keyboard::Scancode::Escape)
                        window.close();
                }
            }
        }

        // Start the game when Enter is pressed
        if (Keyboard::isKeyPressed(Keyboard::Key::Enter) && paused)
        {
            paused = false;

            // Reset the time and the score
            score = 0;
            timeRemaining = 6.0f;
            timeBarWidthPerSecond = timeBarStartWidth / timeRemaining;

            // Make all the branches disappear
            for (int i = 0; i < NUM_BRANCHES; i++)
                branchPositions[i] = side::NONE;

            // Make sure the gravestone is hidden
            spriteRIP.setPosition({ 675.f, 2000.f });

            // Move the player into position
            spritePlayer.setPosition({ 480.f, 600.f });

            acceptInput = true;
        }

        // Player input handling when allowed
        if (acceptInput && !paused)
        {
            if (Keyboard::isKeyPressed(Keyboard::Key::Right))
            {
                playerSide = side::RIGHT;
                score++;

                // Add to the amount of time remaining (guard against division by zero)
                timeRemaining += (2.0f / std::max(1, score)) + 0.15f;
                timeBarWidthPerSecond = timeBarStartWidth / std::max(0.1f, timeRemaining);

                spriteAxe.setPosition({AXE_POSITION_RIGHT, spriteAxe.getPosition().y});
                spritePlayer.setPosition({1000.f, 600.f});

                // update the branches
                updateBranches(score);

                // set the log flying to the left
                spriteLog.setPosition({700.f, 600.f});
                logSpeedX = -5000.f;
                logActive = true;

                acceptInput = false;
                chop.play();
            }

            if (Keyboard::isKeyPressed(Keyboard::Key::Left))
            {
                playerSide = side::LEFT;
                score++;

                // Add to the amount of time remaining
                timeRemaining += (2.0f / std::max(1, score)) + 0.15f;
                timeBarWidthPerSecond = timeBarStartWidth / std::max(0.1f, timeRemaining);

                spriteAxe.setPosition({AXE_POSITION_LEFT, spriteAxe.getPosition().y});
                spritePlayer.setPosition({480.f, 600.f});

                // update the branches
                updateBranches(score);

                // set the log flying
                spriteLog.setPosition({700.f, 600.f});
                logSpeedX = 5000.f;
                logActive = true;

                acceptInput = false;
                chop.play();
            }
        }

        /*
        ****************************************
        Update the scene
        ****************************************
        */
        if (!paused)
        {
            // Measure time
            Time dt = clock.restart();

            // Subtract from the amount of time remaining
            timeRemaining -= dt.asSeconds();

            // avoid negative width
            float newWidth = std::max(0.f, timeBarWidthPerSecond * timeRemaining);
            timeBar.setSize(Vector2f(newWidth, timeBarHeight));

            if (timeRemaining <= 0.0f)
            {
                // Pause the game
                paused = true;

                // Change the message shown to the player
                messageText.setString("Out of time!!");

                // Reposition the text based on its new size
                FloatRect rect = messageText.getLocalBounds();
                messageText.setOrigin({rect.position.x + rect.size.x / 2.0f, rect.position.y + rect.size.y / 2.0f});
                messageText.setPosition({1600.f / 2.0f, 900.f / 2.0f});

                // Play the out of time sound
                outOfTime.play();
            }

            // Setup the bee
            if (!beeActive)
            {
                std::uniform_int_distribution<int> speedDist(200, 399);
                std::uniform_int_distribution<int> heightDist(500, 999);

                beeSpeed = static_cast<float>(speedDist(rng));
                float height = static_cast<float>(heightDist(rng));
                spriteBee.setPosition({2000.f, height});
                beeActive = true;
            }
            else
            {
                // Move the bee
                spriteBee.setPosition({spriteBee.getPosition().x - (beeSpeed * dt.asSeconds()), spriteBee.getPosition().y});

                // Has the bee reached the left hand edge of the screen?
                if (spriteBee.getPosition().x < -100.f)
                {
                    beeActive = false;
                }
            }

            // Manage clouds
            for (int i = 0; i < NUM_CLOUDS; i++)
            {
                if (!cloudsActive[i])
                {
                    std::uniform_int_distribution<int> spdDist(20, 219);
                    std::uniform_int_distribution<int> hDist(0, 149);

                    cloudSpeeds[i] = spdDist(rng);
                    float height = static_cast<float>(hDist(rng));
                    clouds[i].setPosition({-200.f, height});
                    cloudsActive[i] = true;
                }
                else
                {
                    clouds[i].setPosition({clouds[i].getPosition().x + (cloudSpeeds[i] * dt.asSeconds()), clouds[i].getPosition().y});

                    if (clouds[i].getPosition().x > 1600.f)
                    {
                        cloudsActive[i] = false;
                    }
                }
            }

            // Update score/fps text occasionally
            lastDrawn++;
            if (lastDrawn >= 100)
            {
                std::stringstream ss;
                ss << "Score = " << score;
                scoreText.setString(ss.str());

                std::stringstream ss2;
                float fps = 1.0f / std::max(0.0001f, dt.asSeconds());
                ss2 << "FPS = " << static_cast<int>(fps);
                fpsText.setString(ss2.str());

                lastDrawn = 0;
            }

            // update the branch sprites positions and rotations
            for (int i = 0; i < NUM_BRANCHES; i++)
            {
                float height = static_cast<float>(i) * 150.f;

                if (branchPositions[i] == side::LEFT)
                {
                    branches[i].setPosition({520.f, height});
                    branches[i].setOrigin({185.f, 35.f});
                    branches[i].setRotation(sf::degrees(180));
                }
                else if (branchPositions[i] == side::RIGHT)
                {
                    branches[i].setPosition({1110.f, height});
                    branches[i].setOrigin({185.f, 35.f});
                    branches[i].setRotation(sf::degrees(0));
                }
                else
                {
                    branches[i].setPosition({3000.f, height});
                }
            }

            // Handle flying log
            if (logActive)
            {
                spriteLog.setPosition({spriteLog.getPosition().x + (logSpeedX * dt.asSeconds()), spriteLog.getPosition().y + (logSpeedY * dt.asSeconds())});

                if (spriteLog.getPosition().x < -100.f || spriteLog.getPosition().x > 2000.f)
                {
                    logActive = false;
                    spriteLog.setPosition({700.f, 600.f});
                }
            }

            // has the player been squished by a branch?
            if (branchPositions[5] == playerSide)
            {
                // death
                paused = true;
                acceptInput = false;

                // Draw the gravestone
                spriteRIP.setPosition({440.f, 630.f});

                // hide the player
                spritePlayer.setPosition({2000.f, 660.f});

                // Change the text of the message
                messageText.setString("SQUISHED!!");

                // Center it on the screen
                FloatRect rect = messageText.getLocalBounds();
                messageText.setOrigin({rect.position.x + rect.size.x / 2.0f, rect.position.y + rect.size.y / 2.0f});
                messageText.setPosition({1600.f / 2.0f, 900.f / 2.0f});

                // Play the death sound
                death.play();
            }
        } // end if(!paused)

        /*
        ****************************************
        Draw the scene
        ****************************************
        */
        window.clear();

        window.draw(spriteBackground);

        for (int i = 0; i < NUM_CLOUDS; i++)
            window.draw(clouds[i]);

        // Enhanced background trees
        window.draw(spriteTree2);
        window.draw(spriteTree3);
        window.draw(spriteTree4);
        window.draw(spriteTree5);
        window.draw(spriteTree6);

        // Draw branches
        for (int i = 0; i < NUM_BRANCHES; i++)
            window.draw(branches[i]);

        // Draw tree, player, axe, log, gravestone
        window.draw(spriteTree);
        window.draw(spritePlayer);
        window.draw(spriteAxe);
        window.draw(spriteLog);
        window.draw(spriteRIP);

        // UI backgrounds
        window.draw(rect1);
        window.draw(rect2);

        // Bee and UI text
        window.draw(spriteBee);
        window.draw(scoreText);
        window.draw(fpsText);
        window.draw(timeBar);

        if (paused)
            window.draw(messageText);

        window.display();
    }

    return 0;
}

void updateBranches(int seed)
{
    // Move all the branches down one place
    for (int j = NUM_BRANCHES - 1; j > 0; j--) {
        branchPositions[j] = branchPositions[j - 1];
    }

    // Spawn a new branch at position 0
    // LEFT, RIGHT or NONE
    static std::random_device rd;
    static std::mt19937 rng(rd());  // This uses the random device to initialize the RNG.
    rng.seed(seed);  // Set a custom seed

    std::uniform_int_distribution<int> dist(0, 4);
    int r = dist(rng);  // Use the distribution to generate the random number


    // static std::random_device rd;
    // static std::mt19937 rng(rd());
    // std::uniform_int_distribution<int> dist(0, 4);
    // int r = dist(rng + seed); // mix seed into generator call

    switch (r) {
    case 0:
        branchPositions[0] = side::LEFT;
        break;

    case 1:
        branchPositions[0] = side::RIGHT;
        break;

    default:
        branchPositions[0] = side::NONE;
        break;
    }
}