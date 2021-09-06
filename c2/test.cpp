#include <iostream>
#include <string>

void changeVal(int&& ref) {
    ref = 12;
    std::cout << ref << std::endl;
}

int main(void) {
    auto x = 13;
    changeVal(std::move(x));
    std::cout << x << std::endl;

    changeVal(14);
}