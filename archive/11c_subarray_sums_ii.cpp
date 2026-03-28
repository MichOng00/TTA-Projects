#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

int main() {
    int n;
    long long x;
    cin >> n >> x;

    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Compute prefix sums
    vector<long long> prefix(n + 1, 0); // prefix[0] = 0
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + a[i];
    }

    long long ans = 0;

    // Check all subarrays (l, r) using prefix sums
    for (int r = 0; r < n; r++) {
        for (int l = 0; l <= r; l++) {
            // Sum of subarray a[l..r] = prefix[r+1] - prefix[l]
            long long sum_lr = prefix[r + 1] - prefix[l];
            if (sum_lr == x) ans++;
        }
    }


    cout << ans << endl;

}