from collections import defaultdict
from typing import DefaultDict, Tuple, List, Optional, Dict
import sys
import enum


class Solution:
    # https://leetcode-cn.com/problems/utf-8-validation/
    def validUtf8(self, data: List[int]) -> bool:
        remain = 0
        front_mask = 0x00000080  # 1xxxxxxx
        follo_mask = 0x000000C0  # 11xxxxxx
        tripl_mask = 0x000000E0  # 111xxxxx
        quad__mask = 0x000000F0  # 1111xxxx
        ffft__mask = 0x000000F8
        index = 0
        for n in data:
            # print(f'{n:08b}, {remain} ({index})')
            index = index + 1
            if n & front_mask == 0:
                if not remain == 0:
                    return False
            elif n & follo_mask == front_mask:
                if remain <= 0:
                    return False
                else:
                    remain = remain - 1
            else:
                if remain != 0:
                    return False
                if n & ffft__mask == ffft__mask:
                    return False
                elif n & quad__mask == quad__mask:
                    remain = 3
                elif n & tripl_mask == tripl_mask:
                    remain = 2
                elif n & follo_mask == follo_mask:
                    remain = 1
                else:
                    return False
        return remain == 0

    # https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        map1 = DefaultDict(lambda: -1)
        ret = []
        cur_min = sys.maxsize
        for i1, v in enumerate(list1):
            map1[v] = i1
        for i2, v in enumerate(list2):
            i1 = map1[v]
            if i1 >= 0:
                if i1 + i2 < cur_min:
                    ret.clear()
                    ret.append(v)
                    cur_min = i1 + i2
                elif i1 + i2 == cur_min:
                    ret.append(v)
        return ret

    # https://leetcode-cn.com/problems/count-number-of-maximum-bitwise-or-subsets/
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        orMap = defaultdict(lambda: 0)
        for i in range(len(nums)):
            num = nums[i]
            for k, v in list(orMap.items()):
                orVal = num | k
                orMap[orVal] = orMap[orVal] + v
            orMap[num] = orMap[num] + 1

        curMax = -1
        curVal = 0
        for k, v in orMap.items():
            if k > curMax:
                curMax = k
                curVal = v
        return curVal

    def longestWord(self, words: List[str]) -> str:
        wmap = set(words)
        word = ""
        words.sort()
        for w in words:
            if len(word) < len(w):
                match = True
                for l in range(len(w)-1):
                    if not w[0:l+1] in wmap:
                        print(f'break: {w[0:l]}')
                        match = False
                        break
                if match:
                    word = w
        return word

    # https://leetcode-cn.com/problems/the-time-when-the-network-becomes-idle/
    def networkBecomesIdle(self, edges: List[List[int]], patience: List[int]) -> int:
        node_cnt = len(patience)
        dists = [-1] * node_cnt
        dists[0] = 0
        conns = defaultdict(lambda: set())
        for edge in edges:
            conns[edge[0]].add(edge[1])

        q = [0]
        dist = 0
        while (len(q) > 0):
            dist = dist + 1
            q_size = len(q)
            for _ in range(q_size):
                curr = q.pop(0)
                for next in conns[curr]:
                    if -1 == dists[next] or dist < dists[next]:
                        dists[next] = dist
                        q.append(next)
        maxIdle = 0
        for i in range(node_cnt):
            if i > 0:
                curIdle = 0
                if dists[i] * 2 <= patience[i]:
                    curIdle = dists[i] * 2 + 1  # starting from next second
                else:
                    last_send = (dists[i] * 2 - 1) // patience[i] * patience[i]
                    curIdle = last_send + dists[i] * 2 + 1
                maxIdle = max(maxIdle, curIdle)
        return maxIdle

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        if not root:
            return False

        nums = set()
        curr = [root]
        next = []
        while len(curr) > 0:
            for node in curr:
                rm = k - node.val
                if rm in nums:
                    return True
                nums.add(node.val)
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            curr = next
            next = []
        return False

    def findKthNumber5(self, cur: int, ordinal: int, n: int, k: int) -> Tuple[int, int]:
        # cur must end with 0
        gap = 0
        for i in range(0, 10):  # check 0-9
            if cur + i > n:
                return gap, -1
            elif cur + i == 0:
                continue

            # print(f'check {cur + i} ~ {gap + ordinal + 1}')
            gap = gap + 1
            if ordinal + gap == k:
                return -1, cur + i
            else:
                g, v = self.findKthNumber5(10 * (cur + i), ordinal + gap, n, k)
                if g == -1:
                    return -1, v
                else:
                    gap = gap + g
        return gap, -1

    def findKthNumber(self, n: int, k: int) -> int:
        _, val = self.findKthNumber5(0, 0, n, k)
        return val


if __name__ == "__main__":
    s = Solution()
    r = s.findKthNumber(130, 120)
    print(f'result={r}')
