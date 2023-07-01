import json
from typing import List


class Solution:
    # https://leetcode.cn/problems/minimum-size-subarray-sum/
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l = 0
        r = 0
        sum = 0
        min = 0
        while r <= len(nums) and l < len(nums):
            if r > l and sum >= target:
                min = min if min < r - l and min != 0 else r - l
                sum = sum - nums[l]  # remove leftmost number
                l = l + 1
            else:
                if r < len(nums):  # take the next number on the right
                    sum = sum + nums[r]
                r = r + 1

        return min


if __name__ == "__main__":
    s = Solution()
    print(s.minSubArrayLen(nums=[2, 3, 1, 2, 4, 3],   target=700))
