#include <iostream>
#include <string>
#include <vector>

#include "folly/json.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

void runSolution() {
  std::vector<int> profs{1, 2, 3};
  std::vector<int> capts{0, 1, 1};

  sept21::Solution s{};
  auto result = s.findMaximizedCapital(2, 0, profs, capts);
  std::cout << "Result: " << folly::toPrettyJson(result) << std::endl;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::cout << "Input: Code = " << FLAGS_lccn_code << std::endl;
  std::cout << "========= Solution =========" << std::endl;
  runSolution();
  return 0;
}