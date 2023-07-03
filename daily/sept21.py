import heapq
from typing import DefaultDict, List


class Solution:
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None

    def compareVersion(self, version1: str, version2: str) -> int:
        v1s = version1.split(".")
        v2s = version2.split(".")
        len1 = len(v1s)
        len2 = len(v2s)
        maxLen = max(len1, len2)
        for i in range(maxLen):
            s1g = 0 if i >= len1 else int(v1s[i])
            s2g = 0 if i >= len2 else int(v2s[i])
            if s1g != s2g:
                return -1 if s1g < s2g else 1
        return 0

    # https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        sub = head
        for _ in range(k-1):
            if head == None:
                return sub
            else:
                head = head.next

        while head.next != None:
            head = head.next
            sub = sub.next

        return sub

    def smallestK(self, arr: List[int], k: int) -> List[int]:
        maxHeap = []
        if k > 0:
            for n in arr:
                if len(maxHeap) < k:
                    heapq.heappush(maxHeap, -n)
                elif -n > maxHeap[0]:
                    heapq.heappushpop(maxHeap, -n)

        ret = [-n for n in maxHeap]
        return ret

    # https://leetcode-cn.com/problems/implement-rand10-using-rand7/
    def rand10(self):
        fp = 999
        while fp > 6:
            fp = rand7()

        fx = 999
        while fx > 5:
            fx = rand7()

        return fx if fp % 2 == 0 else fx + 5

    def rand7(self):
        return 1


def main():
    s = Solution()
    print(s.smallestK([1, 3, 5, 7, 2, 4, 6, 8], 4))


if __name__ == "__main__":
    main()
