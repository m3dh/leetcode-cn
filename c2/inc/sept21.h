#include <algorithm>
#include <iostream>
#include <queue>
#include <string>

#include "alc.h"
#include "glog/logging.h"

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

  struct Project {
    int profit;
    int capital;
    Project(int profit, int capital) : profit{profit}, capital{capital} {}

    ~Project() {
      cout << "[DTOR] PROF: " << profit << ", CAPT: " << capital << endl;
    }
  };

  // https://leetcode-cn.com/problems/ipo/
  int findMaximizedCapital(int k, int w, vector<int>& profits,
                           vector<int>& capital) {
    // find the affordable best ROI project
    auto profitComp = [](Project& a, Project& b) {
      return a.profit < b.profit;
    };
    auto profitMaxHeap =
        priority_queue<Project, vector<Project>, decltype(profitComp)>{
            profitComp};

    auto projects = vector<Project>{};
    projects.reserve(profits.size());
    for (int i = 0; i < profits.size(); ++i) {
      projects.emplace_back(profits[i], capital[i]);
    }

    auto capComp = [](Project& a, Project& b) { return a.capital < b.capital; };
    sort(projects.begin(), projects.end(), capComp);

    auto projIndex = 0;
    while (k > 0) {
      if (projIndex < projects.size() && projects[projIndex].capital <= w) {
        const Project& proj = projects[projIndex];
        profitMaxHeap.push(move(proj));
        ++projIndex;
      } else {
        if (profitMaxHeap.size() == 0) {
          break;
        } else {
          const Project& proj = profitMaxHeap.top();
          if (proj.profit >= 0) {
            w += proj.profit;
            --k;
          }

          profitMaxHeap.pop();
        }
      }
    }

    return w;
  }
};

}  // namespace sept21