#include <iostream>
#include <string>
#include <vector>

#include "folly/json.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

namespace {
using namespace std;
class Solution {
 public:
  Solution(vector<int>& nums) : nums{nums} {}

  vector<int> reset() { return nums; }

  vector<int> shuffle() {
    vector<int> ret{};
    vector<int> q{nums};
    ret.reserve(nums.size());
    for (int i = 0; i < nums.size(); i++) {
      int idx = std::rand() % q.size();
      auto iter = q.begin();
      std::advance(iter, idx);
      ret.push_back(*iter);
      q.erase(iter);
    }
    return ret;
  }

 private:
  vector<int> nums;
};
}  // namespace

void runSolution() {
  sept21::Solution s{};
  auto result = s.checkValidString("((*)");
  LOG(INFO) << result << std::endl;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  LOG(INFO) << "GLOG Initialized to " << FLAGS_log_dir << std::endl;
  LOG(INFO) << "Input: Code = " << FLAGS_lccn_code << std::endl;
  LOG(INFO) << "========= Solution =========" << std::endl;
  runSolution();
  return 0;
}