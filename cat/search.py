from typing import List, Dict


class Solution:
    # https://leetcode.cn/problems/edit-distance/
    def minDistance(self, word1: str, word2: str) -> int:
        # idea: calculate the sub-distance after matching the first character.
        def minDistanceMemo(w1: str, w2: str, m: Dict[str, int]) -> int:
            if w1 == w2:
                return 0
            elif len(w1) == 0:
                return len(w2)
            elif len(w2) == 0:
                return len(w1)

            # search begin
            key = f'{w1}_{w2}'
            dist = m.get(key)
            if not dist:
                # try 3 different options for the FIRST character.
                # exchange w1 first char to match (or already match)
                dist = minDistanceMemo(
                    w1[1:], w2[1:], m) + (1 if w1[0] != w2[0] else 0)

                # remove w1 first char and try again (without matching!)
                dist = min(dist, minDistanceMemo(w1[1:], w2, m) + 1)

                # add (to w1) to match
                dist = min(dist, minDistanceMemo(w1, w2[1:], m) + 1)

                m[key] = dist
            return dist
        return minDistanceMemo(word1, word2, {})


if __name__ == "__main__":
    s = Solution()
    r = s.minDistance("12334", "1244")
    print(f'result={r}')
