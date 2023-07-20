from typing import List, Tuple, Optional, Set, Dict
from heapq import heappop, heappush
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
    # https://leetcode.cn/problems/maximum-subarray/
    def maxSubArray(self, nums: List[int]) -> int:
        dp = nums.copy()
        for i in range(1, len(dp)):
            dp[i] = max(dp[i], dp[i-1] + dp[i])
        return max(dp)

    # https://leetcode.cn/problems/maximum-sum-circular-subarray/
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        c1 = self.maxSubArray(nums)
        c2 = 0
        n = len(nums)
        r_side_max = [0] * n
        r_side_max[-1] = nums[-1]
        r_sum = nums[-1]
        for i in range(n-2, -1, -1):
            r_sum = nums[i] + r_sum
            r_side_max[i] = max(r_sum, r_side_max[i+1])

        l_sum = 0
        for l in range(n-1):
            l_sum = l_sum + nums[l]
            c2 = max(c2, l_sum + r_side_max[l+1])
        return max(c1, c2)

    # https://leetcode.cn/problems/non-overlapping-intervals/
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # greedy!
        intervals.sort(key=lambda interval: interval[1])
        length = 1
        r = intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] >= r:
                r = intervals[i][1]
                length = length + 1
        return len(intervals) - length

    # https://leetcode.cn/problems/walking-robot-simulation/
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        dir = 0
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # R ===>
        x, y = 0, 0
        maxDist = 0
        obs = set(obstacles)
        for cmd in commands:
            if cmd == -1:
                d = (d + 1) % 4
            elif cmd == -2:
                d = (d + 3) % 4
            else:
                for _ in range(cmd):
                    nx, ny = x + dirs[dir][0], y + dirs[dir][1]
                    if [nx, ny] in obs:
                        break
                    else:
                        x, y = nx, ny
                        maxDist = max(maxDist, x * x + y * y)
        return maxDist

    # https://leetcode.cn/problems/minimum-interval-to-include-each-query/
    def minInterval(
            self, intervals: List[List[int]],
            queries: List[int]) -> List[int]:
        result = [-1] * len(queries)
        intervals.sort(key=lambda interval: interval[0])
        myQueries = [(q, ix) for ix, q in enumerate(queries)]
        myQueries.sort()
        candidates = []
        iix = 0
        for q, ix in myQueries:
            while iix < len(intervals) and intervals[iix][0] <= q:
                intervalLen = intervals[iix][1] - intervals[iix][0] + 1
                heappush(candidates, (intervalLen, intervals[iix]))
                iix = iix + 1
            while len(candidates) > 0 and candidates[0][1][1] < q:
                heappop(candidates)
            if len(candidates) > 0:
                candiLength, candidate = candidates[0]
                if candidate[0] <= q and candidate[1] >= q:
                    result[ix] = candiLength
        return result

    # https://leetcode.cn/problems/add-strings/
    def addStrings(self, num1: str, num2: str) -> str:
        l1 = len(num1)
        l2 = len(num2)
        val: List[str] = []
        c = 0
        for i in range(1, max(l1, l2)+1):
            n1 = num1[l1-i] if l1 - i >= 0 else '0'
            n2 = num2[l2-i] if l2 - i >= 0 else '0'
            n = c + ord(n1) + ord(n2) - ord('0') * 2
            c = n // 10
            n = n % 10
            val.append(n)
        if c > 0:
            val.append(c)
        val.reverse()
        return ''.join(map(str, val))

    # https://leetcode.cn/problems/smallest-sufficient-team/
    def smallestSufficientTeam(
            self, req_skills: List[str],
            people: List[List[str]]) -> List[int]:
        def dfs(
                cur: int, tar: int, pskills: List[int],
                memo: Dict[int, List[int]]) -> List[int]:
            if cur == tar:
                return []

            m = memo.get(cur)
            if m is not None:
                return m
            else:
                sel = None
                nix = -1
                for ix, pmask in enumerate(pskills):
                    neo = cur | pmask
                    if neo != cur:
                        np = dfs(neo, tar, pskills, memo)
                        if sel is None or len(sel) > len(np):
                            sel = np
                            nix = ix
                m = sel.copy()
                m.append(nix)
                memo[cur] = m
                return m

        skill = 1
        skills = {}
        tar_skills = 0
        for s in req_skills:
            tar_skills = tar_skills | skill
            skills[s] = skill
            skill = skill << 1

        pskills: List[int] = [0] * len(people)
        for ix, ps in enumerate(people):
            pmask = 0
            for s in ps:
                if s in skills:
                    pmask = pmask | skills[s]
            pskills[ix] = pmask
        return dfs(0, tar_skills, pskills, {})

    # https://leetcode.cn/problems/longest-arithmetic-subsequence-of-given-difference/
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp: Dict[int, int] = {}
        for n in arr:
            prev = dp.get(n - difference)
            if prev is not None:
                dp[n] = max(prev + 1, dp.get(n) or 1)
            elif n not in dp:
                dp[n] = 1
        return max(dp.values())

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
    r = s.maxSubarraySumCircular(
        [5, -3, 5])
    print(f'Result={json.dumps(r)}')
