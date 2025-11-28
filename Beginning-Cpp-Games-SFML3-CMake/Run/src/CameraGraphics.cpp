#include "CameraGraphics.h"
#include "CameraUpdate.h"
#include "Graphics.cpp"

#include <iostream>
#include <filesystem> // for debugging cwd

CameraGraphics::CameraGraphics(
    RenderWindow* window, Texture* texture,
    Vector2f viewSize, FloatRect viewport)
{
    m_Window = window;
    m_Texture = texture;

    m_View.setSize(viewSize);
    m_View.setViewport(viewport);

    if (viewport.size.x < 1)
    {
        m_IsMiniMap = true;
    }
    else
    {
        m_Font.openFromFile("fonts/KOMIKAP_.ttf");
        m_Text.setFont(m_Font);
        m_Text.setFillColor(Color(255, 0, 0, 255));
        m_Text.setScale({0.2f, 0.2f});
    }

    // Debug: print current working directory
    std::cout << "CWD: " << std::filesystem::current_path() << std::endl;

    if (!m_BackgroundTexture.loadFromFile("graphics/backgroundTexture.png"))
    {
        std::cerr << "Failed to load graphics/backgroundTexture.png\n";
    }
    else
    {
        std::cout << "Loaded background texture: "
                  << m_BackgroundTexture.getSize().x << "x"
                  << m_BackgroundTexture.getSize().y << std::endl;
    }

    m_BackgroundSprite.setTexture(m_BackgroundTexture);
    m_BackgroundSprite2.setTexture(m_BackgroundTexture);

    m_BackgroundSprite.setPosition({0, -200});

    // Prefer explicit extension & check the return value
    bool shaderLoaded = m_Shader.loadFromFile("shaders/glslsandbox109644",
                                              sf::Shader::Type::Fragment);
    if (!shaderLoaded || !m_Shader.isAvailable())
    {
        std::cerr << "Shader failed to load or is not available. path: shaders/glslsandbox109644\n";
    }
    else
    {
        // Set resolution to the view size (or window size) — useful for many shaders
        m_Shader.setUniform("resolution", sf::Vector2f(m_View.getSize()));

        // Bind the background texture to the sampler name your shader uses.
        // If your shader is a Shadertoy port it likely expects "iChannel0".
        // m_Shader.setUniform("iChannel0", m_BackgroundTexture);

        // If your shader samples the texture using a different name, use that name here.
    }

    m_ShaderClock.restart();
}

// #include "CameraGraphics.h"
// #include "CameraUpdate.h"
// #include "Graphics.cpp"

// CameraGraphics::CameraGraphics(
//     RenderWindow* window, Texture* texture,
//     Vector2f viewSize, FloatRect viewport)
// {
//     m_Window = window;
//     m_Texture = texture;

//     m_View.setSize(viewSize);
//     m_View.setViewport(viewport);

//     if (viewport.size.x < 1)
//     {
//         m_IsMiniMap = true;
//     }
//     else
//     {
//         m_Font.openFromFile("fonts/KOMIKAP_.ttf");
//         m_Text.setFont(m_Font);
//         m_Text.setFillColor(Color(255, 0, 0, 255));
//         m_Text.setScale({0.2f, 0.2f});
//     }

//     m_BackgroundTexture.loadFromFile(
//         "graphics/backgroundTexture.png");
//     m_BackgroundSprite.setTexture(m_BackgroundTexture);
//     m_BackgroundSprite2.setTexture(m_BackgroundTexture);

//     m_BackgroundSprite.setPosition({0, -200});

//     m_Shader.loadFromFile(
//         "shaders/glslsandbox109644", sf::Shader::Type::Fragment);

//     if (!m_Shader.isAvailable())
//     {
//         std::cout << "The shader is not available\n";
//     }

//     m_Shader.setUniform("resolution", sf::Vector2f({2500, 2500}));
//     m_ShaderClock.restart();
// }


void CameraGraphics::assemble(
    VertexArray& canvas,
    shared_ptr<Update> genericUpdate,
    IntRect texCoords)
{
    shared_ptr<CameraUpdate> cameraUpdate =
        static_pointer_cast<CameraUpdate>(genericUpdate);
    m_Position = cameraUpdate->getPositionPointer();

    // --- allocate 6 vertices instead of 4 (two triangles) ---
    m_VertexStartIndex = canvas.getVertexCount();
    canvas.resize(canvas.getVertexCount() + 6);

    const float u0 = texCoords.position.x;
    const float v0 = texCoords.position.y;
    const float u1 = texCoords.position.x + texCoords.size.x;
    const float v1 = texCoords.position.y + texCoords.size.y;

    // Triangle 1: 0–1–2
    canvas[m_VertexStartIndex + 0].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 1].texCoords = {u1, v0};
    canvas[m_VertexStartIndex + 2].texCoords = {u1, v1};

    // Triangle 2: 0–2–3
    canvas[m_VertexStartIndex + 3].texCoords = {u0, v0};
    canvas[m_VertexStartIndex + 4].texCoords = {u1, v1};
    canvas[m_VertexStartIndex + 5].texCoords = {u0, v1};
}


float* CameraGraphics::getTimeConnection()
{
    return &m_Time;
}


void CameraGraphics::draw(VertexArray& canvas)
{
    m_View.setCenter(m_Position->position);

    Vector2f startPosition;
    startPosition.x = m_View.getCenter().x - m_View.getSize().x / 2.f;
    startPosition.y = m_View.getCenter().y - m_View.getSize().y / 2.f;

    Vector2f scale;
    scale.x = m_View.getSize().x;
    scale.y = m_View.getSize().y;

    // Quad corners:
    Vector2f p0 = startPosition;
    Vector2f p1 = startPosition + Vector2f(scale.x, 0);
    Vector2f p2 = startPosition + scale;
    Vector2f p3 = startPosition + Vector2f(0, scale.y);

    // --- POSITIONS (triangles) ---
    // Triangle 1
    canvas[m_VertexStartIndex + 0].position = p0;
    canvas[m_VertexStartIndex + 1].position = p1;
    canvas[m_VertexStartIndex + 2].position = p2;

    // Triangle 2
    canvas[m_VertexStartIndex + 3].position = p0;
    canvas[m_VertexStartIndex + 4].position = p2;
    canvas[m_VertexStartIndex + 5].position = p3;

    // --- camera zoom for minimap ---
    if (m_IsMiniMap)
    {
        if (m_View.getSize().x < MAX_WIDTH && m_Position->size.x > 1)
        {
            m_View.zoom(m_Position->size.x);
        }
        else if (m_View.getSize().x > MIN_WIDTH && m_Position->size.x < 1)
        {
            m_View.zoom(m_Position->size.x);
        }
    }

    m_Window->setView(m_View);

    /// --- Parallax background movement ---
    Vector2f movement;
    movement.x = m_Position->position.x - m_PlayersPreviousPosition.x;
    movement.y = m_Position->position.y - m_PlayersPreviousPosition.y;

    if (m_BackgrounsAreFlipped)
    {
        m_BackgroundSprite2.setPosition({
            m_BackgroundSprite2.getPosition().x + movement.x / 6.f,
            m_BackgroundSprite2.getPosition().y + movement.y / 6.f });

        m_BackgroundSprite.setPosition({
            m_BackgroundSprite2.getPosition().x +
            m_BackgroundSprite2.getTextureRect().size.x,
            m_BackgroundSprite2.getPosition().y });

        if (m_Position->position.x >
            m_BackgroundSprite.getPosition().x +
            (m_BackgroundSprite.getTextureRect().size.x / 2.f))
        {
            m_BackgrounsAreFlipped = !m_BackgrounsAreFlipped;
            m_BackgroundSprite2.setPosition(m_BackgroundSprite.getPosition());
        }
    }
    else
    {
        m_BackgroundSprite.setPosition({
            m_BackgroundSprite.getPosition().x - movement.x / 6.f,
            m_BackgroundSprite.getPosition().y + movement.y / 6.f });

        m_BackgroundSprite2.setPosition({
            m_BackgroundSprite.getPosition().x +
            m_BackgroundSprite.getTextureRect().size.x,
            m_BackgroundSprite.getPosition().y });

        if (m_Position->position.x >
            m_BackgroundSprite2.getPosition().x +
            (m_BackgroundSprite2.getTextureRect().size.x / 2.f))
        {
            m_BackgrounsAreFlipped = !m_BackgrounsAreFlipped;
            m_BackgroundSprite.setPosition(m_BackgroundSprite2.getPosition());
        }
    }

    m_PlayersPreviousPosition = m_Position->position;

    // --- Shader uniforms ---
    m_Shader.setUniform("time", m_ShaderClock.getElapsedTime().asSeconds());

    sf::Vector2i mousePos =
        m_Window->mapCoordsToPixel(m_Position->position);

    m_Shader.setUniform("mouse",
        sf::Vector2f(mousePos.x, mousePos.y + 1000));

    if (m_ShaderClock.getElapsedTime().asSeconds() > 10)
    {
        m_ShaderClock.restart();
        m_ShowShader = !m_ShowShader;
    }

    if (!m_ShowShader)
    {
        m_Window->draw(m_BackgroundSprite, &m_Shader);
        m_Window->draw(m_BackgroundSprite2, &m_Shader);
    }
    else
    {
        m_Window->draw(m_BackgroundSprite);
        m_Window->draw(m_BackgroundSprite2);
    }

    // --- UI (only in main camera) ---
    if (!m_IsMiniMap)
    {
        m_Text.setString(std::to_string(m_Time));
        m_Text.setPosition(m_Window->mapPixelToCoords(Vector2i(5, 5)));
        m_Window->draw(m_Text);
    }

    // --- final canvas draw ---
    m_Window->draw(canvas, m_Texture);
}
