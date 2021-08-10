from typing import DefaultDict, List

class Solution:
    # https://leetcode-cn.com/problems/super-ugly-number/
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

    # https://leetcode-cn.com/problems/arithmetic-slices/
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        cnt = 0
        cur = 0 # current length
        stp = 0 # step
        prv = 0 # previous number
        nums.append(9998877)
        for num in nums:
            if cur == 0:
                cur = 1
                prv = num
            elif cur == 1:
                cur = 2
                stp = num - prv
                prv = num
            elif num - prv == stp:
                prv = num
                cur = cur + 1
            else:
                if cur > 2:
                    # 3 - 1, 4 - 3, 5 - 6 => (cur - 2 + 1) * (cur - 2) / 2
                    cnt = cnt + (cur - 1) * (cur - 2) / 2
                stp = num - prv
                prv = num
                cur = 2

        return int(cnt)

    # https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        total = 0
        dp = [DefaultDict(int) for _ in nums]
        for i, x in enumerate(nums):
            for j in range(i):
                delta = x - nums[j]
                precnt = dp[j][delta]
                newcnt = precnt + 1 # adding a new len=2 slice
                total = total + precnt # len >= 3
                dp[i][delta] = dp[i][delta] + newcnt

        return total

def main():
    s = Solution()
    print(s.numberOfArithmeticSlices([1,2,3,4,1,2,3]))

if __name__ == "__main__":
    main()