from typing import List, Tuple


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        sortedNums: List[Tuple[int, int]] = []
        for i, n in enumerate(nums):
            sortedNums.append((i, n))
        sortedNums.sort(key=lambda t: t[1])
        l = 0
        r = len(sortedNums) - 1
        while l < r:
            cur = sortedNums[l][1] + sortedNums[r][1]
            if cur < target:
                l = l + 1
            elif cur > target:
                r = r - 1
            else:
                return [sortedNums[l][0], sortedNums[r][0]]
        return [-1, -1]


if __name__ == "__main__":
    s = Solution()
    r = s.twoSum([1, 2, 3, 4], 5)
    print(f'Result={r}')
