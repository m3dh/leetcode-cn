#include <queue>
#include <string>

#include "alc.h"

namespace sept21 {
using namespace std;
using namespace alc;

// the solution class for programming quiz's in Sept. 2021
class Solution {
 public:
  // https://leetcode-cn.com/problems/sum-of-left-leaves/
  int sumOfLeftLeaves(TreeNode* root) {
    int sum = 0;
    queue<TreeNode*> level{};
    if (root != nullptr) {
      // root cannot be a left leaf node.
      level.push(root);
    }

    while (level.size() > 0) {
      int size = level.size();
      for (int i = 0; i < size; ++i) {
        TreeNode* node = level.front();
        level.pop();
        if (node->left != nullptr) {
          if (node->left->left == nullptr && node->left->right == nullptr) {
            sum += node->left->val;
          }
          level.push(node->left);
        }

        if (node->right != nullptr) {
          level.push(node->right);
        }
      }
    }

    return sum;
  }

  // https://leetcode-cn.com/problems/split-a-string-in-balanced-strings/
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

}  // namespace sept21