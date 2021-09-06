#include <iostream>
#include <string>
#include "gflags/gflags.h"
#include "inc/sept21.h"

DEFINE_int32(lccn_code, 0, "A useless input code");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Code = " << FLAGS_lccn_code << std::endl;

    Sept21::Solution s {};
    auto result = s.balancedStringSplit(std::string("LLRRLRLR"));
    std::cout << "Result: " << result << std::endl;

    return 0;
}