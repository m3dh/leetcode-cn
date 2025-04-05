# Leetcode Project in Apr. 2025
import os
from typing import List


class Solution:
    # https://leetcode.cn/problems/largest-divisible-subset/?envType=daily-question&envId=2025-04-06
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums = sorted(nums, reverse=True)
        dp = [1] * len(nums)
        mx = -1
        me = 0
        for e in range(1, len(nums)):
            for s in range(0, e):
                if nums[s] % nums[e] == 0:
                    dp[e] = max(dp[e], dp[s] + 1)
                    if dp[e] > mx:
                        mx = dp[e]
                        me = e
        ret = [nums[me]]
        ce = me
        cm = mx - 1
        for i in range(me - 1, -1, -1):
            if nums[i] % nums[ce] == 0 and dp[i] == cm:
                ret.append(nums[i])
                cm = cm - 1
                ce = i
        return ret


if __name__ == "__main__":
    s = Solution()
    r = s.largestDivisibleSubset([1, 2, 3, 4, 8])
    print(",".join([str(i) for i in r]))
