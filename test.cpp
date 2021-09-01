#include <iostream>
#include <string>

std::string getObject(std::string s) {
    std::string ret = s + " -> obj";
    std::cout << "init: " << s << std::endl;
    return ret;
}

void func(std::string s) {
    const static std::string v = getObject(s);
    std::cout << " >> " << v << " << " << std::endl;
}

int main(void) {
    func("daily");
    func("daily");
    func("weekly");
    func("yyely");
}