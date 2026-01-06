from typing import List, Set, Optional
from functools import cmp_to_key
import heapq


class Solution:
    def bestClosingTime(self, customers: str) -> int:
        loss_after = [0] * (len(customers) + 1)
        for i in range(len(customers) - 1, -1, -1):
            loss_after[i] = 1 if customers[i] == "Y" else 0
            if i < len(customers) - 1:
                loss_after[i] += loss_after[i + 1]
        min_loss = -1
        best_pos = -1
        acc_cust = 0
        for i in range(len(customers) + 1):
            close_loss = acc_cust + (loss_after[i] if i < len(customers) else 0)
            if close_loss < min_loss or min_loss == -1:
                min_loss = close_loss
                best_pos = i
            if i < len(customers) and customers[i] == "N":
                acc_cust = acc_cust + 1
        return best_pos

    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        sorted_capacity = sorted(capacity, reverse=True)
        cix = 0
        remain = 0
        for a in apple:
            while remain < a and cix < len(sorted_capacity):
                remain = remain + sorted_capacity[cix]
                cix += 1
            if remain < a:
                return -1
            else:
                remain -= a
        return cix

    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        def frac_compare(a: List[int], b: List[int]):
            am = a[0] * b[1]
            bm = b[0] * a[1]
            return am - bm

        vals = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                vals.append([arr[i], arr[j]])
        return vals.sort(key=cmp_to_key(frac_compare))[k - 1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        arr = [0] * (amount + 1)
        for coin in coins:
            for i in range(len(arr) - coin):
                if i == 0 or arr[i] > 0:
                    arr[i + coin] = (
                        arr[i] + 1
                        if arr[i + coin] == 0
                        else min(arr[i] + 1, arr[i + coin])
                    )
        return arr[amount]

    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        rooms_avl = []
        rooms_cnt = []
        rooms_use = []
        for i in range(n):
            rooms_avl.append(i)
            rooms_cnt.append(0)

        # heap 1: least available meeting room #
        heapq.heapify(rooms_avl)

        # heap 2: least end meeting time
        # heapq.heapify(rooms_use)

        cur_time = 0
        for meeting in sorted(meetings):
            # sorted meetings by start time
            if cur_time < meeting[0]:
                cur_time = meeting[0]

            # check the delay case
            if len(rooms_avl) == 0:
                # this will pop the first end meeting room and with smallest ix (if multiple meetings end)
                end = heapq.heappop(rooms_use)
                cur_time = max(cur_time, end[0])
                heapq.heappush(rooms_avl, end[1])

            # try finish some more meetings
            while len(rooms_use) > 0 and rooms_use[0][0] <= cur_time:
                end = heapq.heappop(rooms_use)
                heapq.heappush(rooms_avl, end[1])

            if len(rooms_avl) > 0:
                # find a meeting room
                room = heapq.heappop(rooms_avl)
                rooms_cnt[room] = rooms_cnt[room] + 1
                end_time = (
                    meeting[1]
                    if meeting[0] >= cur_time
                    else meeting[1] + (cur_time - meeting[0])
                )
                # print(
                #    f"Meeting: {meeting[0]} -> {meeting[1]} in {room}, end: {end_time} (cur: {cur_time})"
                # )
                heapq.heappush(rooms_use, [end_time, room])
            else:
                raise RuntimeError("Unable to find available meeting room")

        mx, ix = 0, -1
        for i, cnt in enumerate(rooms_cnt):
            if cnt > mx:
                mx = cnt
                ix = i
        return ix

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ret = {}
        for str in strs:
            cnt = [0] * 26
            for ch in str:
                cnt[ord(ch) - ord("a")] += 1
            ret.setdefault(tuple(cnt), []).append(str)
        return [val for _, val in ret.items()]

    def moveZeroes(self, nums: List[int]) -> None:
        wp = 0
        rp = 0
        while rp < len(nums):
            if nums[rp] == 0:
                rp = rp + 1
            else:
                if rp != wp:
                    nums[wp] = nums[rp]
                rp = rp + 1
                wp = wp + 1
        while wp < len(nums):
            nums[wp] = 0
            wp = wp + 1

    def minPathSum(self, grid: List[List[int]]) -> int:
        memo = []
        for row in grid:
            memo.append([-1] * len(row))

        def dfsMinPath(x: int, y: int, memo: List[List[int]]) -> int:
            nonlocal grid
            if memo[x][y] != -1:
                return memo[x][y]

            fr, fd = -1, -1
            if x + 1 < len(grid):
                fr = grid[x][y] + dfsMinPath(x + 1, y, memo)
            if y + 1 < len(grid[x]):
                fd = grid[x][y] + dfsMinPath(x, y + 1, memo)

            if fr == -1 and fd == -1:
                memo[x][y] = grid[x][y]
            else:
                memo[x][y] = min(
                    fr if fr != -1 else float("inf"), fd if fd != -1 else float("inf")
                )
            return memo[x][y]

        return dfsMinPath(0, 0, memo)

    def countNegatives(self, grid: List[List[int]]) -> int:
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0

        x, y, cnt = 0, len(grid[0]) - 1, 0

        # find the first neg in each row or end (no negative)
        # move left and down
        while x < len(grid):
            while y > 0 and grid[x][y - 1] < 0:
                y = y - 1  # y: [0, len(grid-1)]

            if grid[x][y] < 0:
                cnt = cnt + len(grid[x]) - y

            x = x + 1

        return cnt

    def solveNQueens(self, n: int) -> List[List[str]]:
        ret = []

        def dfs(cur: List[int], s: int) -> None:
            if s == n:
                sol = []
                for c in cur:
                    line = []
                    for i in range(n):
                        if c == i:
                            line.append("Q")
                        else:
                            line.append(".")
                    sol.append("".join(line))
                ret.append(sol)
            else:
                for i in range(n):
                    valid = True
                    for j, p in enumerate(cur):
                        if p == i:
                            valid = False
                            break
                        elif i + (s - j) == p:
                            valid = False
                            break
                        elif i - (s - j) == p:
                            valid = False
                            break
                    if valid:
                        cur.append(i)
                        dfs(cur, s + 1)
                        cur.pop()

        dfs([], 0)
        return ret

    def longestValidParentheses(self, s: str) -> int:
        # n
        ans = 0
        stk = [-1]
        for i in range(len(s)):
            if s[i] == "(":
                stk.append(i)
            else:
                if len(stk) > 0:
                    stk.pop()

                if len(stk) > 0:
                    ans = max(ans, i - stk[-1])
                else:
                    stk.append(i)

        return ans

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        ret = []
        if len(words) == 0 or len(s) < len(words[0]) * len(words):
            return ret

        def update_and_check_count(word_cnt, word, num) -> bool:
            if word in word_cnt:
                word_cnt[word] += num
            else:
                word_cnt[word] = num

            if word_cnt[word] == 0:
                del word_cnt[word]

            return len(word_cnt) == 0

        # the key idea is to window the word length as well!
        word_len = len(words[0])
        full_len = len(words) * word_len
        for i in range(word_len):
            l = i
            r = i
            word_cnt = {}  # idea: expected words start with cnt = -1
            for word in words:
                if word in word_cnt:
                    word_cnt[word] -= 1
                else:
                    word_cnt[word] = -1

            while r < len(s):
                print(f"{i}, {l}, {r}")
                curr_len = r - l
                if curr_len < full_len:
                    # take one word in
                    r = r + word_len
                    if r - 1 < len(s):
                        word = s[r - word_len : r]
                        curr_len = r - l
                        # print(f"I={i}, In: {word} ({r})")
                        if (
                            update_and_check_count(word_cnt, word, 1)
                            and curr_len == full_len
                        ):
                            ret.append(l)

                print(f"{curr_len}, {full_len}")
                if curr_len == full_len:
                    # take one word out
                    l = l + word_len
                    word = s[l - word_len : l]
                    # print(f"I={i}, Out: {word} ({l})")
                    update_and_check_count(word_cnt, word, -1)
        return ret

    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        allowed_map = {}
        for allow in allowed:
            k = allow[:2]
            v = allow[2:]
            allowed_map.setdefault(k, []).append(v)

        def dmffr(cur: str, nxt: List[str], results: Set[str], ix: int) -> None:
            if ix == len(cur) - 1:
                results.add("".join(nxt))
            else:
                k = "".join(cur[ix : ix + 2])
                if vals := allowed_map.get(k):
                    for val in vals:
                        nxt[ix] = val
                        dmffr(cur, nxt, results, ix + 1)

        def dmffs(b: str, failed: Set[str]) -> bool:
            if len(b) == 1:
                return True
            nxt = [""] * (len(b) - 1)
            nxs = set()
            dmffr(b, nxt, nxs, 0)
            for nx in nxs:
                # print(f"From {b} -> {nx}")
                if nx not in failed:
                    su = dmffs(nx, failed)
                    if su:
                        return True
                    failed.add(nx)
            return False

        return dmffs(bottom, set())

    def rotate(self, nums: List[int], k: int) -> None:
        for _ in range(k):
            prev = nums[-1]
            for i in range(nums):
                cur = nums[i]
                nums[i] = prev
                prev = cur

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        decending_queue = []  # index queue
        results = []
        for i in range(len(nums)):
            # 0, 1, 2, 3 / k = 2
            #       ^
            while len(decending_queue) > 0 and decending_queue[0] <= i - k:
                # pop numbers out of the window
                decending_queue.pop(0)

            while len(decending_queue) > 0 and nums[decending_queue[-1]] <= nums[i]:
                # pop numbers that are smaller (overriden)
                decending_queue.pop(-1)

            decending_queue.append(i)
            if i >= k - 1:
                results.append(nums[decending_queue[0]])

        return results

    def findAnagrams(self, s, p):
        ctr = {}
        ix = []
        for pc in p:
            if pcnt := ctr.get(pc):
                ctr[pc] = pcnt - 1
            else:
                ctr[pc] = -1
        l, r = 0, 0
        while r < len(s):
            nx = s[r]
            r = r + 1
            if nxcnt := ctr.get(nx):
                if nxcnt == -1:
                    del ctr[nx]  # eq 0
                else:
                    ctr[nx] = nxcnt + 1
            else:
                ctr[nx] = 1

            if r - l == len(p):
                if len(ctr) == 0:
                    ix.append(l)
                px = s[l]
                l = l + 1
                if pxcnt := ctr.get(px):
                    if pxcnt == 1:
                        del ctr[px]  # eq 0
                    else:
                        ctr[px] = pxcnt - 1
                else:
                    ctr[px] = -1
        return ix

    def subarraySum(self, nums: List[int], k: int) -> int:
        sum_count = {0: 1}
        run_sum = 0
        total = 0
        for num in nums:
            run_sum = run_sum + num
            extra = run_sum - k
            if cnt := sum_count.get(extra):
                total = total + cnt
            if cur_cnt := sum_count.get(run_sum):
                sum_count[run_sum] = cur_cnt + 1
            else:
                sum_count[run_sum] = 1
        return total

    def minWindow(self, s: str, t: str) -> str:
        target = {}
        for cn in t:
            if cn in target:
                target[cn] += 1
            else:
                target[cn] = 1

        l, r = 0, -1
        actual = {}
        minStr = None
        while True:
            met = all(cn in actual and actual[cn] >= cnt for cn, cnt in target.items())
            if met:
                if minStr is None or r - l + 1 < len(minStr):
                    minStr = s[l : r + 1]
                ln = s[l]
                l = l + 1
                actual[ln] -= 1
            else:
                if r + 1 >= len(s):
                    break
                rn = s[r + 1]
                r = r + 1
                if rn in actual:
                    actual[rn] += 1
                else:
                    actual[rn] = 1
        return minStr

    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        total_day = len(cells) + 1
        final_map = [[total_day] * col for _ in range(row)]
        for day, cell in enumerate(cells):
            # before day + 1, cell is still unblocked...
            final_map[cell[0] - 1][cell[1] - 1] = day + 1

        def bfs(day: int) -> bool:
            q = [(0, c) for c in range(col) if final_map[0][c] > day]
            visited = set(q)
            while len(q) > 0:
                top = q.pop(0)
                # print(f"day:{day}, visit:{top}")
                if top[0] == row - 1:
                    return True
                else:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = dr + top[0], dc + top[1]
                        if nr >= 0 and nc >= 0 and nr < row and nc < col:
                            if final_map[nr][nc] > day:
                                npos = (nr, nc)
                                if npos not in visited:
                                    visited.add(npos)
                                    q.append(npos)
            return False

        # do binary day search
        lower, upper = 0, total_day
        result = 0
        while lower <= upper:
            mid = (lower + upper) // 2
            if bfs(mid):
                result = mid
                lower = mid + 1
            else:
                upper = mid - 1
        return result

    def decodeString(self, s: str) -> str:
        stk = ["1", "["]
        cur = []
        for ch in s + "]":
            if ch == "[":
                stk.append("".join(cur))
                stk.append("[")
                cur = []
            elif ch == "]":
                if len(cur) > 0:
                    stk.append("".join(cur))
                    cur = []
                s = stk.pop()
                while stk[-1] != "[":
                    s = stk.pop() + s
                assert stk.pop() == "["
                n = stk.pop()
                stk.append(int(n) * s)
            else:
                if len(cur) > 0 and not str(cur[-1]).isdigit() and ch.isdigit():
                    stk.append("".join(cur))
                    cur = []
                cur.append(ch)
        return stk[0]

    def sumFourDivisors(self, nums: List[int]) -> int:
        ret = 0
        for num in nums:
            fac_sum = 0
            fac_cnt = 0
            fac = 1
            while fac * fac <= num:
                if num % fac == 0:
                    fac_cnt += 1
                    fac_sum += fac
                    if num // fac != fac:
                        fac_cnt += 1
                        fac_sum += num // fac
                fac = fac + 1
            if fac_cnt == 4:
                ret += fac_sum
        return ret

    def repeatedNTimes(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            for j in range(1, 4):
                if i + j < len(nums) and nums[i + j] == nums[i]:
                    return nums[i]

        return -1

    def generateParenthesis(self, n: int) -> List[str]:
        ret = []
        cur = []

        def dfs(cur, ret, l, r) -> None:
            if l == 0 and r == 0:
                ret.append("".join(cur))
            else:
                if l > 0:
                    cur.append("(")
                    dfs(cur, ret, l - 1, r)
                    cur.pop()
                if r > 0 and r - 1 >= l:
                    cur.append(")")
                    dfs(cur, ret, l, r - 1)
                    cur.pop()

        dfs(cur, ret, n, n)
        return ret

    def longestDupSubstring(self, s: str) -> str:
        def findDupSubstring(s: str, n: int) -> str | None:
            subs = set()
            # s=[abcde] 5, n=3, 0,1,2
            for i in range(len(s) - n + 1):
                ss = s[i : i + n]  # 0, 3
                if ss in subs:
                    return ss
                else:
                    subs.add(ss)
            return None

        l, r = 1, len(s)
        mss = ""
        while l <= r:
            mid = (l + r) // 2
            ss = findDupSubstring(s, mid)
            if ss and len(ss) > len(mss):
                mss = ss
                l = mid + 1
            else:
                r = mid - 1
        return mss

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def maxProduct(self, root: Optional[TreeNode]) -> int:
        nodeSums = set()

        def getSubTreeSum(root: Optional[TreeNode], is_root: bool) -> int:
            if root is None:
                return 0
            ret = (
                getSubTreeSum(root.left, False)
                + getSubTreeSum(root.right, False)
                + root.val
            )
            if not is_root:
                nodeSums.add(ret)
            return ret

        mx = 0
        rootSum = getSubTreeSum(root, True)
        for v in nodeSums:
            if v * (rootSum - v) > mx:
                mx = v * (rootSum - v)
        return mx % 1000000007


if __name__ == "__main__":
    s = Solution()
    r = s.generateParenthesis(3)
    print(f"r ==> {r}")
