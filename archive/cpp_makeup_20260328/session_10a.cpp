#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

vector <long long> psum(const vector<int> &arr){
    vector<long long> psums(arr.size() + 1);
    psums[0] = 0;
    // for (int i = 0;  i < arr.size(); i++){ psums[i + 1] + psums[i] + arr[i]; }
    partial_sum(arr.begin(), arr.end(), psums.begin() + 1);
    return psums;
}

int main(){
    int N, Q;
    cin >> N >> Q;
    vector<int> nums(N);
    for (int i = 0; i < N; i++){cin >> nums[i];}

    vector<long long> prefix_arr = psum(nums);

    for (int i = 0; i < Q; i++){
        int l, r;
        cin >> l >> r;
        cout << prefix_arr[r] - prefix_arr[l] << endl;
    }
}