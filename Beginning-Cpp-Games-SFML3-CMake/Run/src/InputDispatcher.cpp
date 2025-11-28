#include "InputDispatcher.h"
#include "InputReceiver.h"

InputDispatcher::InputDispatcher(sf::RenderWindow* window)
{
    m_Window = window;
}

void InputDispatcher::dispatchInputEvents()
{
    // SFML 3: pollEvent returns std::optional<sf::Event>
    while (const std::optional<sf::Event> event = m_Window->pollEvent())
    {
        // Dispatch event to all receivers
        for (auto& ir : m_InputReceivers)
        {
            ir->addEvent(*event);   // addEvent takes a concrete sf::Event
        }
    }
}

void InputDispatcher::registerNewInputReceiver(InputReceiver* ir)
{
    m_InputReceivers.push_back(ir);
}
