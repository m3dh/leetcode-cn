from typing import DefaultDict, List

class Solution:
    # https://leetcode-cn.com/problems/boats-to-save-people/
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        # greedy...
        cnt = 0
        l = 0
        r = len(people) - 1
        people.sort()
        while l <= r:
            if r == l or people[l] + people[r] > limit:
                r = r - 1
                cnt = cnt + 1
            else:
                l = l + 1
                r = r - 1
                cnt = cnt + 1
        return cnt

def main():
    s = Solution()
    print(s.numRescueBoats([1,2,3,4], 4))

if __name__ == "__main__":
    main()