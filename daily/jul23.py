from typing import List, Tuple, Optional
from heapq import heapify, heappop, heappush
from enum import Enum


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            sum = numbers[l] + numbers[r]
            if sum < target:
                l = l + 1
            elif sum == target:
                return [l+1, r+1]
            else:
                r = r - 1
        return []

    def findSolution(self, customfunction: 'CustomFunction', z: int) -> List[List[int]]:
        ret: List[List[int]] = []
        y = 1000
        for x in range(1, 1001):
            while y >= 1 and customfunction.f(x, y) > z:
                y = y - 1
            if y == 0:
                break
            elif customfunction.f(x, y) == z:
                ret.append([x, y])
        return ret

    # https://leetcode.cn/problems/remove-k-digits/
    def removeKdigits(self, num: str, k: int) -> str:
        stack: List[str] = []
        for n in num:
            while k > 0 and len(stack) > 0 and stack[-1] > n:
                stack.pop()  # pop last element
                k = k - 1
            stack.append(n)
        while k > 0:
            stack.pop()
            k = k - 1
        for i, n in enumerate(stack):
            if n != '0':
                return ''.join(stack[i:])
        return '0'

    # https://leetcode.cn/problems/sum-in-a-matrix/
    def matrixSum(self, nums: List[List[int]]) -> int:
        for l in nums:
            # sort descending
            l.sort(reverse=True)
        score = 0
        for i in range(len(nums[0])):
            curMax = 0
            for j in range(len(nums)):
                # find cur max from all rows
                if nums[j][i] > curMax:
                    curMax = nums[j][i]
            score = score + curMax
        return score

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

    # https://leetcode.cn/problems/add-two-numbers/
    def addTwoNumbers1(
            self, l1: Optional[ListNode],
            l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(val=-1, next=ListNode(val=0))
        prev = dummy
        remaining = 0
        while l1 != None or l2 != None or remaining != 0:
            v1 = 0
            v2 = 0
            if l1 != None:
                v1 = l1.val
                l1 = l1.next
            if l2 != None:
                v2 = l2.val
                l2 = l2.next
            curVal = remaining + v1 + v2
            remaining = 1 if curVal >= 10 else 0
            curVal = curVal if curVal < 10 else curVal - 10
            prev.next = ListNode(val=curVal)
            prev = prev.next
        return dummy.next

    # https://leetcode.cn/problems/add-two-numbers-ii/
    def addTwoNumbers2(
            self, l1: Optional[ListNode],
            l2: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(l: Optional[ListNode]) -> Optional[ListNode]:
            prev: Optional[ListNode] = None
            if not l:
                return l
            while True:
                next = l.next
                l.next = prev
                prev = l
                if next:
                    l = next
                else:
                    return l
        r1 = reverse(l1)
        r2 = reverse(l2)
        ret = self.addTwoNumbers1(r1, r2)
        return reverse(ret)


if __name__ == "__main__":
    s = Solution()
    r = s.removeKdigits("1001", 1)
    print(f'Result={r}')
