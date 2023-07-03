from typing import List, Tuple, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
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
    r = s.twoSum([1, 2, 3, 4], 5)
    print(f'Result={r}')
