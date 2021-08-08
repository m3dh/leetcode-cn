# https://leetcode-cn.com/problems/super-ugly-number/

from typing import List

class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        memo = [0] * n
        memo[0] = 1
        pcnt = len(primes)
        pidx = [0] * pcnt
        for i in range(1, n):
            min_val = min(memo[pidx[j]] * primes[j] for j in range(pcnt))
            memo[i] = min_val
            # print(f'memo i={i}, val={min_val}')
            for j in range(pcnt):
                if memo[pidx[j]] * primes[j] == min_val:
                    pidx[j] = pidx[j] + 1

        return memo[n-1]

def main():
    s = Solution()
    print(s.nthSuperUglyNumber(300, [2,3,7]))

if __name__ == "__main__":
    main()