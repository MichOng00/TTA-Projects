#include <iostream>

void printHello() // if fn does not return anything, use void
{
    std::cout << "Hello from function!\n";
    // return; // can be used just like Python to exit early
}
int rtnfive()
{
    return 5;
}

int main() // indentation and whitespace do not matter
{
    // print
    std::cout << "Hello World!\n"; // new std::cout does not newline

    /////////// VARIABLES - MUST declare type
    int x;
	int y, z; // multiple declarations (same type)

	//std::cout << x; // this would be an error 
	x = 5; // assignment 
    std::cout << x << "\n"; // could do single quotes
    std::cout << "x is equal to: " << x;

    ////////// INITIALIZATION
    int a;         // default-initialization (no initializer)

    // Traditional initialization forms:
    int b = 5;     // copy-initialization (initial value after equals sign)
    int c(6);   // direct-initialization (initial value in parenthesis)

    // Modern initialization forms (preferred):
    int d{ 7 };   // direct-list-initialization (initial value in braces)
    int e{};      // value-initialization (will be 0 or other default)
	// benefit of d is that it prevents narrowing conversions

    ////////// INPUT
	std::cout << "\nPlease enter a number: ";
	std::cin >> x; // input from keyboard
    std::cout << "You entered " << x << '\n';
    // if not integer, fails and x stays as 0
    // try: h, 3.2, 1000000000000, 123abc, abc123, (x=5)
    // also affects later inputs

    std::cout << "Enter two numbers separated by a space: ";
    std::cin >> x >> y; // get two numbers and store in variable x and y respectively
    std::cout << "You entered " << x << " and " << y << '\n';

    ////////// FUNCTIONS (see above - note cannot nest defns)
    printHello();
	std::cout << "rtnfive() returns: " << rtnfive() << '\n';



	return 0; // main must return an int
}
