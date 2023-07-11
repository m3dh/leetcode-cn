from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None or headB is None:
            return None

        a = headA
        b = headB
        while True:
            if a is None and b is None:
                return None
            elif a is None:
                a = headB
                b = b.next
            elif b is None:
                b = headA
                a = a.next
            elif a == b:
                return a
            else:
                a = a.next
                b = b.next

    # https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
    def removeNthFromEnd(
            self, head: Optional[ListNode],
            n: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        f = dummy
        s = dummy
        for _ in range(n):
            f = f.next

        while True:
            if f.next is None:
                s.next = s.next.next
                return dummy.next
            else:
                f = f.next
                s = s.next
