#include <iostream>
#define N 8
int main()
{
    for (int i = 0; i < N; i++)
    {
        std::string str = std::to_string(i % (N >> 1));
        std::cout << str << std::endl;
    }

    for (int i = 0; i > -N; i--)
    {
        std::string str = std::to_string(i % (N >> 1));
        std::cout << str << std::endl;
    }
}