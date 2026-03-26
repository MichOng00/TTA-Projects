#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

int main(){
    int N; // number of cows in the vector
    int Q; // number of queries to make
    cin >> N >> Q;

    // Make 3 different vectors to hold each type of cow
    vector<int> holsteins(N + 1);
    vector<int> guernseys(N + 1);
    vector<int> jerseys(N + 1);

    for (int c = 0; c < N; c++){
        holsteins[c + 1] = holsteins[c];
        guernseys[c + 1] = guernseys[c];
        jerseys[c + 1] = jerseys[c];

        int type;
        cin >> type;
        if (type == 1) holsteins[c + 1]++;
        else if (type == 2) guernseys[c + 1]++;
        else if (type == 3) jerseys[c + 1]++;
    }

    for (int q = 0; q < Q; q++){
        int start;
        int end;
        cin >> start >> end;
        cout << holsteins[end] - holsteins[start - 1] << ' '
             << guernseys[end] - guernseys[start - 1] << ' '
             << jerseys[end] - jerseys[start - 1] << '\n';
    }
}