from typing import DefaultDict, List


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


if __name__ == "__main__":
    s = Solution()
