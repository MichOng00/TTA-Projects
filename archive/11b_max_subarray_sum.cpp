//https://cses.fi/problemset/task/1643
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

int main() {
    int n;
    cin >> n;

    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    long long prefix = 0;        // current prefix sum
    long long min_prefix = 0;    // smallest prefix seen so far
    long long ans = LLONG_MIN;   // maximum subarray sum

    for (int i = 0; i < n; i++) {
        prefix += a[i] ;

        // Try taking subarray ending at i
        ans = max(ans, prefix - min_prefix);

        // Update smallest prefix
        min_prefix = min(min_prefix, prefix);
    }

    cout << ans << endl;

}
