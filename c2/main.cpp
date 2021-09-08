#include <iostream>
#include <string>
#include <vector>

#include "folly/json.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

void runSolution() {
  std::vector<std::string> words{
      "This", "is", "an", "example", "of", "text", "justification."};
  sept21::Solution s{};
  auto result = s.fullJustify(words, 16);
  for (auto line : result) {
    std::cout << "[L]: " << line << std::endl;
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::cout << "Input: Code = " << FLAGS_lccn_code << std::endl;
  std::cout << "========= Solution =========" << std::endl;
  runSolution();
  return 0;
}