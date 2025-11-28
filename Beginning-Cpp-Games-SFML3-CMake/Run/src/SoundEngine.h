#pragma once
#include <SFML/Audio.hpp>

using namespace sf;

class SoundEngine
{
private:
    // Singleton instance
    static SoundEngine* m_s_Instance;

    // Static music & flags
    static Music music;

    // --- Buffers must remain static and shared ---
    static SoundBuffer m_ClickBuffer;
    static SoundBuffer m_JumpBuffer;
    static SoundBuffer m_FireballLaunchBuffer;

    // --- Sounds must NOT be static (SFML 3 requirement) ---
    Sound m_ClickSound;
    Sound m_JumpSound;
    Sound m_FireballLaunchSound;

public:
    SoundEngine();

	static bool mMusicIsPlaying;

    // Music interface
    static void startMusic();
    static void pauseMusic();
    static void resumeMusic();
    static void stopMusic();

    // SFX interface
    static void playClick();
    static void playJump();
    static void playFireballLaunch(
        Vector2f playerPosition,
        Vector2f soundLocation);
};

// Declare global instance (defined in .cpp)
extern SoundEngine g_SoundEngine;
