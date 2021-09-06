#include <iostream>
#include <string>
#include "sept21.h"

int main(void) {
    Sept21::Solution s {};
    auto result = s.balancedStringSplit(std::string("LLRRLRLR"));
    std::cout << "Result: " << result << std::endl;

    return 0;
}