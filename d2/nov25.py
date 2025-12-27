from typing import List, Optional, Tuple
from json import dumps


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        l = len(colors)
        r = 0
        ix = 0
        while ix < l:
            curr = colors[ix]
            mx = neededTime[ix]
            cur_sum = neededTime[ix]
            ix = ix + 1
            while ix < l and colors[ix] == curr:
                mx = max(mx, neededTime[ix])
                cur_sum = cur_sum + neededTime[ix]
                ix = ix + 1
            cur_sum = cur_sum - mx
            r = r + cur_sum
        return r

    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        ret = [False] * len(nums)
        prefix = 0
        for i in range(len(nums)):
            prefix = prefix * 2 + nums[i]
            prefix = prefix % 5
            ret[i] = prefix == 0
        return ret
