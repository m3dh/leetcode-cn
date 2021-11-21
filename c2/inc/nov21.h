#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>

#include "alc.h"

namespace nov21 {
using namespace std;
using namespace alc;

// the solution class for programming quiz's in Nov. 2021
class Solution {
 public:
  int maxDepth(Node* root) {
    if (root == nullptr) {
      return 0;
    } else {
      int max = 0;
      if (root->children.size() > 0) {
        for (Node* child : root->children) {
          max = std::max(max, maxDepth(child));
        }
      }

      return max + 1;
    }
  }

  int lengthOfLongestSubstring(string s) {
    std::set<char> set{};
    int l = 0;
    int m = 0;
    for (int r = 0; r < s.length(); r++) {
      char c = s[r];
      if (set.find(c) != set.end()) {
        while (l < r) {
          char rm = s[l];
          set.erase(rm);
          l++;
          if (rm == c) break;
        }
        set.insert(c);
      } else {
        set.insert(c);
      }

      m = std::max(m, (int)set.size());
    }

    return m;
  }
};

}  // namespace nov21