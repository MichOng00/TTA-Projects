#include "InputReceiver.h"
#include "InputDispatcher.h"

void InputReceiver::addEvent(Event event)
{
    mEvents.push_back(event);
}

vector<Event>& InputReceiver::getEvents()
{
    return mEvents;
}

void InputReceiver::clearEvents()
{
    mEvents.clear();
}
