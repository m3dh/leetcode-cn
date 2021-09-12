#include <iostream>
#include <string>
#include <vector>

#include "folly/json.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

void runSolution() {
  sept21::Solution s{};
  auto result = s.checkValidString("((*)");
  LOG(INFO) << result << std::endl;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  LOG(INFO) << "GLOG Initialized to " << FLAGS_log_dir << std::endl;
  LOG(INFO) << "Input: Code = " << FLAGS_lccn_code << std::endl;
  LOG(INFO) << "========= Solution =========" << std::endl;
  runSolution();
  return 0;
}