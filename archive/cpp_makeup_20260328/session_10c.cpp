#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

const int maxN = 1e5 + 10;

int main(){
    int arr[maxN];
    int prefGcd[maxN]; // prefGcd[i] = GCD of a1, a2, ..., ai
    int suffGcd[maxN]; // suffGcd[i] = GCD of ai, ai+1, ..., an
    int N;

    cin >> N;
    for (int i = 1; i <= N; i++) cin >> arr[i];

    prefGcd[0] = 0;
    suffGcd[N + 1] = 0;

    for (int i = 1; i <= N; i++) { prefGcd[i] = gcd(prefGcd[i-1], arr[i]); }

    for (int i = N; i >= 1; i--) { suffGcd[i] = gcd(suffGcd[i+1], arr[i]); }

    int ans = 0;
    for (int i = 1; i <= N; i++){
        ans = max(ans, gcd(prefGcd[i - 1], suffGcd[i + 1]));
    }
    cout << ans << endl;
}