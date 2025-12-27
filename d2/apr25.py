# Leetcode Project in Apr. 2025
import os
from typing import List, Optional, Tuple
import json


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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

    # https://leetcode.cn/problems/lowest-common-ancestor-of-deepest-leaves/?envType=daily-question&envId=2025-04-04
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def ldlRec(root: TreeNode) -> Tuple[TreeNode, int]:
            if root is None:
                return (None, 0)
            elif root.left is None and root.right is None:
                return (root, 1)

            ln, ld = ldlRec(root.left)
            rn, rd = ldlRec(root.right)
            if ld == rd:
                return (root, ld + 1)
            elif ld > rd:
                return (ln, ld + 1)
            else:
                return (rn, rd + 1)

        n, _ = ldlRec(root)
        return n

    def canPartition(self, nums: List[int]) -> bool:
        sum_nums = sum(nums)
        if sum_nums % 2 == 1:
            return False

        target = (int)(sum_nums / 2)
        dp = [False] * (target + 1)
        dp[0] = True
        known_max = 0
        for n in nums:
            for l in range(known_max, -1, -1):
                if dp[l] and l + n <= target:
                    dp[l + n] = True
                    known_max = max(known_max, l + n)
        return known_max == target

    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        wx = 0
        cx = 0
        cn = 0
        cv = nums[0] - 1
        for cx in range(len(nums)):
            if nums[cx] == cv:
                cn = cn + 1
                if cn <= 2:
                    nums[wx] = cv
                    wx = wx + 1
            else:
                cn = 1
                cv = nums[cx]
                nums[wx] = cv
                wx = wx + 1
        return wx

    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        kmin = 0
        kmax = 0
        cur = 0
        for d in differences:
            cur = cur + d
            kmin = min(kmin, cur)
            kmax = max(kmax, cur)
        diff = kmax - kmin
        tdiff = upper - lower
        return 0 if diff > tdiff else (tdiff - diff + 1)

    # https://leetcode.cn/problems/count-subarrays-with-score-less-than-k
    def countSubarrays(self, nums: List[int], k: int) -> int:
        l, r = 0, 0
        cnt, sum, subLen = 0, 0, 0
        while r < len(nums):
            sum = sum + nums[r]
            r = r + 1  # move to next
            subLen = subLen + 1
            while sum * subLen >= k and l < r:
                sum = sum - nums[l]
                l = l + 1
                subLen = subLen - 1
            if sum * subLen < k:
                cnt = cnt + subLen
                # print(f"{l}, {r}, {subLen}, {sum}, {cnt}")
        return cnt

    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        if len(mat) == 0:
            return []
        n = len(mat)
        m = len(mat[0])
        ret = []
        rev = False
        s = 0
        while True:
            fnd = False
            i = s
            j = 0
            loc = []
            while i >= 0:
                if i >= 0 and i < n and j < m:
                    loc.append(mat[i][j])
                    fnd = True
                i = i - 1
                j = j + 1
            if len(loc) > 0:
                ret.extend(reversed(loc) if rev else loc)
                rev = not rev

            if not fnd:
                break
            else:
                s = s + 1
        return ret

    # VB4NIUN95X
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        dp = [0] * len(nums)
        for i in range(len(nums)):
            n = 0 if i == 0 else dp[i - 1]
            s = 0 if i <= 1 else dp[i - 2]
            dp[i] = max(n, s + nums[i])
        return max(dp[-1], dp[-2])


if __name__ == "__main__":
    s = Solution()
    r = s.rob([1, 2, 3, 1])
    print(f"r ==> {r}")
