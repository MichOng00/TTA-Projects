#include "SoundEngine.h"
#include <assert.h>

// -------------------------------------------------
// Static members (safe: buffers + music)
// -------------------------------------------------
SoundEngine* SoundEngine::m_s_Instance = nullptr;

bool SoundEngine::mMusicIsPlaying = false;

Music SoundEngine::music;

SoundBuffer SoundEngine::m_ClickBuffer;
SoundBuffer SoundEngine::m_JumpBuffer;
SoundBuffer SoundEngine::m_FireballLaunchBuffer;


// -------------------------------------------------
// Constructor (creates all Sound instances normally)
// -------------------------------------------------
SoundEngine::SoundEngine()
    : m_ClickSound(m_ClickBuffer),
      m_JumpSound(m_JumpBuffer),
      m_FireballLaunchSound(m_FireballLaunchBuffer)
{
    assert(m_s_Instance == nullptr);
    m_s_Instance = this;

    // Load buffers (safe: buffers are static)
    m_ClickBuffer.loadFromFile("sound/click.wav");
    m_JumpBuffer.loadFromFile("sound/jump.wav");
    m_FireballLaunchBuffer.loadFromFile("sound/fireballLaunch.wav");

    // Audio listener setup
    Listener::setDirection({1.f, 0.f, 0.f});
    Listener::setUpVector({1.f, 1.f, 0.f});
    Listener::setGlobalVolume(100.f);
}


// -------------------------------------------------
// Sound effect functions
// -------------------------------------------------
void SoundEngine::playClick()
{
    m_s_Instance->m_ClickSound.play();
}

void SoundEngine::playJump()
{
    m_s_Instance->m_JumpSound.play();
}

void SoundEngine::playFireballLaunch(
    Vector2f playerPosition,
    Vector2f soundLocation)
{
    Sound& snd = m_s_Instance->m_FireballLaunchSound;

    snd.setRelativeToListener(true);

    if (playerPosition.x > soundLocation.x)
    {
        Listener::setPosition({0.f, 0.f, 0.f});
        snd.setPosition({-100.f, 0.f, 0.f});
    }
    else
    {
        Listener::setPosition({0.f, 0.f, 0.f});
        snd.setPosition({100.f, 0.f, 0.f});
    }

    snd.setMinDistance(100.f);
    snd.setAttenuation(0.f);
    snd.play();
}


// -------------------------------------------------
// Music control
// -------------------------------------------------
void SoundEngine::startMusic()
{
    music.openFromFile("music/music.wav");
    music.play();
    music.setLooping(true);
    mMusicIsPlaying = true;
}

void SoundEngine::pauseMusic()
{
    music.pause();
    mMusicIsPlaying = false;
}

void SoundEngine::resumeMusic()
{
    music.play();
    mMusicIsPlaying = true;
}

void SoundEngine::stopMusic()
{
    music.stop();
    mMusicIsPlaying = false;
}

SoundEngine g_SoundEngine;
