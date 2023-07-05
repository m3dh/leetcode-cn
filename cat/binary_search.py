from typing import List


class Solution:
    # https://leetcode.cn/problems/find-positive-integer-solution-for-a-given-equation/
    def findSolution(self, customfunction: 'CustomFunction', z: int) -> List[List[int]]:
        ret: List[List[int]] = []
        for x in range(1, 1001):
            if customfunction.f(x, 1) <= z and customfunction.f(x, 1000) >= z:
                # bi-search y
                l, r = 1, 1000
                while l <= r:
                    m = (l + r) // 2
                    val = customfunction.f(x, m)
                    if val == z:
                        ret.append([x, m])
                        break
                    elif val > z:
                        r = m - 1
                    else:
                        l = m + 1
        return ret

    # https://leetcode.cn/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 注意：这是一个模板，用 <=, m-1, m+1 可以保证最后 l 是小于 t 的最大数
        # 因为只有在 nums[m] < target 的时候 l 才会被赋值
        l = 0
        r = len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m - 1  # ---> r >= t
            else:
                l = m + 1  # ---> l <= t
        return l + 1

    # https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def searchLeftSide(tt: int) -> int:
            l = 0
            r = len(nums) - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] >= tt:
                    r = m - 1
                else:
                    l = m + 1
            return l  # as result, l is LE tt
        a = searchLeftSide(target)
        b = searchLeftSide(target + 1)
        if a < len(nums) and nums[a] == target:
            return [a, b-1]
        else:
            return [-1, -1]


if __name__ == "__main__":
    pass
