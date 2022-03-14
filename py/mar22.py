from collections import defaultdict
import enum
from typing import DefaultDict, List
import sys


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


if __name__ == "__main__":
    s = Solution()
    r = s.countMaxOrSubsets([1, 3])
    print(f'result={r}')
