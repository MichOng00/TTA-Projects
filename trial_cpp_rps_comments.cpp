/*
To assess:
1. Main function and return type
2. Print something (include iostream); use namespace
3. Create a variable (rand and srand with cstdlib - not so important)
4. Get user input and store it
5. Create and initialize an array (const keyword)
6. Logic for tie/win/lose (in that order)
7. Validate input (basic, advanced) - tests edge case thinking
8. Play again logic
0. Can you find a way to break the program? (enter y) - then fix
*/

#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    srand(time(nullptr));

    int userChoice;
    int computerChoice;

    char playAgain = 'y';

    cout << "Rock Paper Scissors\n";

    while (playAgain == 'y' || playAgain == 'Y')
    {

        cout << "Enter your choice:\n";
        cout << "0 = Rock\n1 = Paper\n2 = Scissors\n";
        cout << "Your choice: ";
        //cin >> userChoice; // before 7. validation

        // Check non-int input
        if (!(cin >> userChoice))
        {
            cout << "Please enter a number (0, 1, or 2).\n";
            // 9. Fix corner case - clear input buffer
            //cin.clear();
            //cin.ignore(1000, '\n');
            continue; // return 0 without loop
        }

        // Check input is valid option
        if (userChoice < 0 || userChoice > 2)
        {
            cout << "Number must be 0, 1, or 2.\n";
            continue; // return 0 without loop
        }

        computerChoice = rand() % 3;

        const string choices[3] = { "Rock", "Paper", "Scissors" };

        cout << "You chose: " << choices[userChoice] << "\n";
        cout << "Computer chose: " << choices[computerChoice] << "\n";

        if (userChoice == computerChoice)
        {
            cout << "It's a tie!\n";
        }
        else if (
            (userChoice == 0 && computerChoice == 2) ||
            (userChoice == 1 && computerChoice == 0) ||
            (userChoice == 2 && computerChoice == 1)
            )
        {
            cout << "You win!\n";
        }
        else
        {
            cout << "You lose!\n";
        }

        cout << "\nPlay again? (y/n): ";
        cin >> playAgain;
    }

    cout << "\nThanks for playing!\n";

    return 0;
}
