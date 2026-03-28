// https://cses.fi/problemset/task/1662
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Compute prefix sums mod n, size n+1 so that prefix[0] = 0 (base case)
    // prefix[i] represents the sum of a[0..i-1] modulo n
    vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = (prefix[i] + a[i]) % n;
        // Fix negative remainders (C++ can return negative results for % with negative values)
        if (prefix[i + 1] < 0) prefix[i + 1] += n;
    }

    // freq[r] = how many prefix sums so far have remainder r (mod n)
    vector<int> freq(n, 0);
    long long ans = 0;

    for (int i = 0; i <= n; i++) {
        // If prefix[i] has appeared before, each previous occurrence forms
        // a valid subarray ending at i whose sum is divisible by n
        ans += freq[prefix[i]];
        // Record this remainder for future subarrays
        freq[prefix[i]]++;
    }

    cout << ans << endl;
}