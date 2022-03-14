#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "folly/json.h"
#include "folly/experimental/coro/Task.h"
#include "folly/experimental/coro/BlockingWait.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

class Test {
  public:
  Test() {
    LOG(INFO) << "Default CTOR";
  }
  Test(const Test& t) {
    LOG(INFO) << "Default COPY-CTOR";
  }
  Test(Test&& t) {
    LOG(INFO) << "Default MOVE-CTOR";
  }
  folly::coro::Task<void> func() {
    co_return;
  }
};

int main(int argc, char **argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  LOG(INFO) << "GLOG Initialized to " << FLAGS_log_dir << std::endl;
  LOG(INFO) << "Input: Code = " << FLAGS_lccn_code << std::endl;
  auto up = std::make_unique<Test>();
  auto up1 = std::unique_ptr<Test>(std::move(up));
  folly::coro::blockingWait(up1->func());
  return 0;
}