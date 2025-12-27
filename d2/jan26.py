from typing import List
from functools import cmp_to_key


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
        import heapq

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


if __name__ == "__main__":
    s = Solution()
    r = s.mostBooked(3, [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]])
    print(f"r ==> {r}")
