from typing import DefaultDict, List

class Solution:
    # https://leetcode-cn.com/problems/super-ugly-number/
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        memo = [0] * n
        memo[0] = 1
        pcnt = len(primes)
        pidx = [0] * pcnt
        for i in range(1, n):
            min_val = min(memo[pidx[j]] * primes[j] for j in range(pcnt))
            memo[i] = min_val
            # print(f'memo i={i}, val={min_val}')
            for j in range(pcnt):
                if memo[pidx[j]] * primes[j] == min_val:
                    pidx[j] = pidx[j] + 1

        return memo[n-1]

    # https://leetcode-cn.com/problems/arithmetic-slices/
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        cnt = 0
        cur = 0 # current length
        stp = 0 # step
        prv = 0 # previous number
        nums.append(9998877)
        for num in nums:
            if cur == 0:
                cur = 1
                prv = num
            elif cur == 1:
                cur = 2
                stp = num - prv
                prv = num
            elif num - prv == stp:
                prv = num
                cur = cur + 1
            else:
                if cur > 2:
                    # 3 - 1, 4 - 3, 5 - 6 => (cur - 2 + 1) * (cur - 2) / 2
                    cnt = cnt + (cur - 1) * (cur - 2) / 2
                stp = num - prv
                prv = num
                cur = 2

        return int(cnt)

    # https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        total = 0
        dp = [DefaultDict(int) for _ in nums]
        for i, x in enumerate(nums):
            for j in range(i):
                delta = x - nums[j]
                precnt = dp[j][delta]
                newcnt = precnt + 1 # adding a new len=2 slice
                total = total + precnt # len >= 3
                dp[i][delta] = dp[i][delta] + newcnt

        return total

    # https://leetcode-cn.com/problems/longest-palindromic-subsequence/
    def longestPalindromeSubseq(self, s: str) -> int:
        l = len(s)
        memo = [[0] * l for _ in range(l)]
        return self.lps516Rec(s, 0, l-1, memo)

    def lps516Rec(self, s: str, l: int, r: int, memo: List[List[int]]) -> int:
        if l > r: return 0
        elif l == r: return 1
        elif r - l == 1:
            if s[l] == s[r]: return 2
            else: return 1
        elif memo[l][r] != 0: return memo[l][r]
        else:
            mLen = 0
            if s[l] == s[r]:
                mLen = self.lps516Rec(s, l+1, r-1, memo) + 2

            mLen = max(mLen, self.lps516Rec(s, l+1, r, memo), self.lps516Rec(s, l, r-1, memo))
            memo[l][r] = mLen
            return mLen

    # https://leetcode-cn.com/problems/number-of-digit-one/
    def countDigitOne(self, n: int) -> int:
        # Note: we do count duplications!
        k, mulk = 0, 1
        ans = 0
        while n >= mulk:
            ans += (n // (mulk * 10)) * mulk + min(max(n % (mulk * 10) - mulk + 1, 0), mulk)
            k += 1
            mulk *= 10
        return ans

def main():
    s = Solution()
    print(s.longestPalindromeSubseq("abcdef"))

if __name__ == "__main__":
    main()