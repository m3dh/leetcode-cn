#include <string>

namespace Sept21 {
using namespace std;

// the solution class for programming quiz's in Sept. 2021
class Solution {
public:
    int balancedStringSplit(string s) {
        int l = 0;
        int r = 0;
        int cnt = 0;
        for (auto ch : s) {
            if (ch == 'L') {
                ++l;
            } else {
                ++r;
            }

            if (l == r) {
                ++cnt;
            }
        }

        return cnt;
    }
};

}