#include <iostream>
using namespace std;

int main() {
    char letter;
    cout << "Enter a lowercase letter: ";
    cin >> letter;

    switch(letter) {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
            cout << letter << " is a vowel.\n";
            break;
        default:
            cout << letter << " is a consonant.\n";
    }

    // char grade;
	// cout << "Enter your grade (A-F): ";
    // cin >> grade;
    // switch(grade)
    // {
    // case 'A':
    //     cout << "Excellent!";
    //     break;
    // case 'B':
    //     cout << "Good work.";
    //     break;
    // case 'E':
    // case 'F':
    //     cout << "You failed, try again next time!";
    //     break;
    // default:
    //     cout << "Not the worst, but you can do better.";
    // }

    // return 0;

    return 0;
}
