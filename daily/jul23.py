from typing import List, Tuple, Optional, Set, Dict
from heapq import heapify, heappop, heappush
from enum import Enum
import json


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # https://leetcode.cn/problems/sum-of-distances-in-tree/
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        # bottom-up
        def collect(
                cur: int, paths: List[List[int]],
                colls: List[List[int]],
                parent: int) -> List[int]:
            sum = 0
            cnt = 1  # self
            for nxt in paths[cur]:
                if parent != nxt:
                    nxtCollect = collect(nxt, paths, colls, cur)
                    cnt = cnt + nxtCollect[1]
                    sum = sum + nxtCollect[0] + nxtCollect[1]
            colls[cur] = [sum, cnt]
            return colls[cur]

        # top-down
        def deploy(  # up = [sum, cnt]
                cur: int, up: List[int], paths: List[List[int]],
                colls: List[List[int]],
                res: List[int],
                parent: int) -> None:
            res[cur] = up[0] + up[1] + colls[cur][0]  # up_sum + up_cnt
            for nxt in paths[cur]:
                if parent != nxt:
                    upsum = colls[cur][0] - colls[nxt][0] - colls[nxt][1] + up[0] + up[1]
                    upcnt = colls[cur][1] - colls[nxt][1] + up[1]
                    nup = [upsum, upcnt]
                    deploy(nxt, nup, paths, colls, res, cur)

        # edges -> paths
        paths: List[List[int]] = [[] for _ in range(n)]
        colls: List[List[int]] = [[0, 0] for _ in range(n)]  # [sum, cnt]
        res: List[int] = [0] * n
        for edge in edges:
            paths[edge[0]].append(edge[1])
            paths[edge[1]].append(edge[0])

        collect(0, paths, colls, -1)
        deploy(0, [0, 0], paths, colls, res, -1)
        return res

    # https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended-ii/
    def maxValue(self, events: List[List[int]], k: int) -> int:
        def maxValRec(
                events: List[List[int]],
                k: int, ix: int, memo: List[List[int]]) -> int:
            if k == 0 or ix >= len(events):
                return 0
            maxVal = memo[ix][k]
            if maxVal >= 0:
                return maxVal
            else:
                endDay = events[ix][1]

                # don't pick
                maxVal = max(events[ix][2], maxValRec(events, k, ix + 1, memo))

                # pick it
                l, r = ix + 1, len(events) - 1
                while l <= r:
                    m = (l + r) // 2
                    if events[m][0] <= endDay:
                        l = l + 1
                    else:
                        r = r - 1
                maxVal = max(maxVal, events[ix][2] +
                             maxValRec(events, k - 1, l, memo))
                memo[ix][k] = maxVal
                return maxVal
        memo = [[-1] * (k+1) for _ in range(len(events))]
        events.sort(key=lambda ev: ev[0])
        return maxValRec(events, k, 0, memo)

    # https://leetcode.cn/problems/find-eventual-safe-states/
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        def isSafe(
                ix: int, term: List[bool],
                safe: Dict[int, bool],
                visited: Set[int]) -> bool:
            nonlocal graph
            if term[ix]:
                return True
            elif ix in safe and safe[ix]:
                return safe[ix]
            elif ix in visited:
                safe[ix] = False
                return False
            else:
                visited.add(ix)
                ret = all(
                    [isSafe(nx, term, safe, visited) for nx in graph[ix]])
                safe[ix] = ret
                return ret

        term = [len(g) == 0 for g in enumerate(graph)]
        safe: Dict[int, bool] = {}
        ret = [i for i in range(len(graph)) if isSafe(i, term, safe, set())]
        return ret

    # https://leetcode.cn/problems/distribute-coins-in-binary-tree/
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        count = 0

        def traverse(root: TreeNode) -> int:
            nonlocal count
            l = traverse(root.left) if root.left is not None else 0
            r = traverse(root.right) if root.right is not None else 0
            remain = l + r + root.val - 1
            count = count + abs(l) + abs(r)
            return remain
        if root is not None:
            traverse(root)
        return count

    # https://leetcode.cn/problems/course-schedule/
    def canFinish(
            self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        inDegree: List[Set[int]] = [set() for _ in range(numCourses)]
        nextCourses: List[Set[int]] = [set() for _ in range(numCourses)]
        for pair in prerequisites:
            cur, pre = pair
            inDegree[cur].add(pre)
            nextCourses[pre].add(cur)

        queue: List[int] = []
        for cur, ind in enumerate(inDegree):
            if len(ind) == 0:
                queue.append(cur)

        while len(queue) > 0:
            cur = queue.pop(0)
            for nextCourse in nextCourses[cur]:
                ind: Set[int] = inDegree[nextCourse]
                ind.remove(cur)
                if len(ind) == 0:
                    queue.append(nextCourse)
        return all([len(ind) == 0 for ind in inDegree])

    # https://leetcode.cn/problems/minimum-falling-path-sum/
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        dp = [[0] * n for _ in range(n)]
        imax = 10 ** 20
        for x in range(n):
            for y in range(n):
                prev = dp[x-1][y] if x > 0 else 0
                prev = min(prev, dp[x-1][y-1] if x > 0 and y > 0 else imax)
                prev = min(prev, dp[x-1][y+1] if x > 0 and y < n - 1 else imax)
                dp[x][y] = matrix[x][y] + prev
        return min(dp[-1])

    # https://leetcode.cn/problems/alternating-digit-sum/
    def alternateDigitSum(self, n: int) -> int:
        nums = []
        while n > 0:
            num = n % 10
            n = n // 10
            nums.append(num)
        nums.reverse()
        ret = 0
        sg = 1
        for num in nums:
            ret = ret + num * sg
            sg = sg * -1
        return ret

    # https://leetcode.cn/problems/3sum-closest/
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if len(nums) < 3:
            return -1
        nums.sort()
        ret = -1
        delta = 10**10
        for i in range(0, len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                ld = abs(target-sum)  # local delta
                if ld < delta:
                    delta = ld
                    ret = sum
                if sum == target:
                    return ret
                elif sum < target:
                    l = l + 1
                else:
                    r = r - 1
        return ret

    # https://leetcode.cn/problems/3sum/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        ret = []
        nums.sort()
        for i in range(0, len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum == 0:
                    ret.append([nums[i], nums[l], nums[r]])
                    r = r - 1

                    while l < r and nums[l] == nums[l-1]:
                        l = l + 1
                    while l < r and nums[r] == nums[r + 1]:
                        r = r - 1
                elif sum < 0:
                    l = l + 1
                else:
                    r = r - 1
        return ret

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
    r = s.sumOfDistancesInTree(
        n=2, edges=[[0, 1]])
    print(f'Result={json.dumps(r)}')
