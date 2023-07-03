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

    def generateMatrix(self, n: int) -> List[List[int]]:
        l = 0
        r = n - 1
        u = 0
        b = n - 1
        num = 1
        target = n * n
        ret: List[List[int]] = [[0] * n for _ in range(n)]
        while num <= target:
            for i in range(l, r + 1):  # 'til r
                ret[u][i] = num
                num = num + 1
            u = u + 1  # shrink upper bound

            for i in range(u, b + 1):  # 'til b
                ret[i][r] = num
                num = num + 1
            r = r - 1  # shrink right bound

            for i in range(r, l - 1, -1):
                ret[b][i] = num
                num = num + 1
            b = b - 1  # shrink bottom bound

            for i in range(b, u - 1, -1):
                ret[i][l] = num
                num = num + 1
            l = l + 1
        return ret


if __name__ == "__main__":
    s = Solution()
    print(json.dumps(s.generateMatrix(9)))
