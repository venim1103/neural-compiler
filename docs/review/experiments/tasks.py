"""Twelve small integer-result tasks typical of tool-integrated reasoning.

Each task has: a natural-language prompt, an idiomatic Python solution, a plain
(loop-only, no bignum/stdlib) Python solution, and a freestanding C `solve()`
that compiles for both x86-64 and wasm32 (no libc).
"""

SENTENCE = "the quick brown fox jumps over the lazy dog while seven wizards quietly hex jumbo bags"

TASKS = {
    "primes_sum": dict(
        prompt="Compute the sum of all prime numbers below 20000.",
        py_idiomatic="""N = 20000
s = bytearray([1]) * N
s[0] = s[1] = 0
for i in range(2, int(N**0.5) + 1):
    if s[i]:
        s[i*i::i] = bytearray(len(range(i*i, N, i)))
print(sum(i for i in range(N) if s[i]))
""",
        py_plain="""def solve():
    n = 20000
    is_p = [True] * n
    total = 0
    for i in range(2, n):
        if is_p[i]:
            total += i
            for j in range(i * i, n, i):
                is_p[j] = False
    return total
print(solve())
""",
        c="""static unsigned char comp[20000];
long long solve(void) {
    long long total = 0;
    for (int i = 2; i < 20000; i++) {
        if (!comp[i]) {
            total += i;
            for (int j = i * i; j < 20000; j += i) comp[j] = 1;
        }
    }
    return total;
}
""",
    ),
    "collatz": dict(
        prompt="Among starting values 1 <= s < 10000, find the s whose Collatz sequence takes the most steps to reach 1 (smallest s on ties).",
        py_idiomatic="""def steps(n):
    c = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        c += 1
    return c
print(max(range(1, 10000), key=steps))
""",
        py_plain="""def solve():
    best, best_len = 1, 0
    for s in range(1, 10000):
        n, length = s, 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            length += 1
        if length > best_len:
            best_len, best = length, s
    return best
print(solve())
""",
        c="""long long solve(void) {
    long long best = 1, best_len = 0;
    for (long long s = 1; s < 10000; s++) {
        long long n = s, len = 0;
        while (n != 1) {
            n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
            len++;
        }
        if (len > best_len) { best_len = len; best = s; }
    }
    return best;
}
""",
    ),
    "lcm_1_25": dict(
        prompt="Compute the least common multiple of all integers from 1 to 25.",
        py_idiomatic="""import math
print(math.lcm(*range(1, 26)))
""",
        py_plain="""def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def solve():
    r = 1
    for k in range(1, 26):
        r = r // gcd(r, k) * k
    return r
print(solve())
""",
        c="""static long long gcd(long long a, long long b) {
    while (b) { long long t = a % b; a = b; b = t; }
    return a;
}
long long solve(void) {
    long long r = 1;
    for (long long k = 1; k <= 25; k++) r = r / gcd(r, k) * k;
    return r;
}
""",
    ),
    "fib_mod": dict(
        prompt="Compute the 80th Fibonacci number modulo 1000000007, with F(0)=0 and F(1)=1.",
        py_idiomatic="""a, b = 0, 1
for _ in range(80):
    a, b = b, (a + b) % 1000000007
print(a)
""",
        py_plain="""def solve():
    a, b = 0, 1
    for _ in range(80):
        a, b = b, (a + b) % 1000000007
    return a
print(solve())
""",
        c="""static volatile int n_iter = 80; /* volatile: stop compile-time folding */
long long solve(void) {
    long long a = 0, b = 1;
    for (int i = 0; i < n_iter; i++) {
        long long t = (a + b) % 1000000007;
        a = b; b = t;
    }
    return a;
}
""",
    ),
    "divisors": dict(
        prompt="Compute the number of positive divisors of 9699690.",
        py_idiomatic="""n = 9699690
print(sum(1 if d * d == n else 2 for d in range(1, int(n**0.5) + 1) if n % d == 0))
""",
        py_plain="""def solve():
    n = 9699690
    count = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            count += 1 if d * d == n else 2
        d += 1
    return count
print(solve())
""",
        c="""long long solve(void) {
    long long n = 9699690, count = 0;
    for (long long d = 1; d * d <= n; d++)
        if (n % d == 0) count += (d * d == n) ? 1 : 2;
    return count;
}
""",
    ),
    "binomial": dict(
        prompt="Compute the binomial coefficient C(40, 20).",
        py_idiomatic="""import math
print(math.comb(40, 20))
""",
        py_plain="""def solve():
    r = 1
    for i in range(1, 21):
        r = r * (20 + i) // i
    return r
print(solve())
""",
        c="""static volatile long long k = 20; /* volatile: stop compile-time folding */
long long solve(void) {
    long long r = 1;
    for (long long i = 1; i <= k; i++) r = r * (k + i) / i;
    return r;
}
""",
    ),
    "digits_2_100": dict(
        prompt="Compute the sum of the decimal digits of 2 to the power 100.",
        py_idiomatic="""print(sum(map(int, str(2**100))))
""",
        py_plain="""def solve():
    digits = [1]
    for _ in range(100):
        carry = 0
        for i in range(len(digits)):
            v = digits[i] * 2 + carry
            digits[i] = v % 10
            carry = v // 10
        if carry:
            digits.append(carry)
    return sum(digits)
print(solve())
""",
        c="""static int dig[64];
long long solve(void) {
    int len = 1;
    dig[0] = 1;
    for (int k = 0; k < 100; k++) {
        int carry = 0;
        for (int i = 0; i < len; i++) {
            int v = dig[i] * 2 + carry;
            dig[i] = v % 10;
            carry = v / 10;
        }
        if (carry) dig[len++] = carry;
    }
    long long s = 0;
    for (int i = 0; i < len; i++) s += dig[i];
    return s;
}
""",
    ),
    "modpow": dict(
        prompt="Compute 3 to the power 1000000000, modulo 1000000007.",
        py_idiomatic="""print(pow(3, 10**9, 10**9 + 7))
""",
        py_plain="""def solve():
    m = 1000000007
    base, e, r = 3, 1000000000, 1
    while e > 0:
        if e & 1:
            r = r * base % m
        base = base * base % m
        e >>= 1
    return r
print(solve())
""",
        c="""long long solve(void) {
    const long long m = 1000000007;
    long long base = 3, e = 1000000000, r = 1;
    while (e > 0) {
        if (e & 1) r = r * base % m;
        base = base * base % m;
        e >>= 1;
    }
    return r;
}
""",
    ),
    "lcg_median": dict(
        prompt="Let x0 = 42 and x(k+1) = (1103515245*x(k) + 12345) mod 2^31. Take x1..x51, sort them, and return the median (the 26th smallest).",
        py_idiomatic="""x, xs = 42, []
for _ in range(51):
    x = (1103515245 * x + 12345) % 2**31
    xs.append(x)
print(sorted(xs)[25])
""",
        py_plain="""def solve():
    a = []
    x = 42
    for _ in range(51):
        x = (1103515245 * x + 12345) % 2147483648
        a.append(x)
    for i in range(1, 51):
        v = a[i]
        j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v
    return a[25]
print(solve())
""",
        c="""static long long a[51];
long long solve(void) {
    long long x = 42;
    for (int i = 0; i < 51; i++) {
        x = (1103515245LL * x + 12345) % 2147483648LL;
        a[i] = x;
    }
    for (int i = 1; i < 51; i++) {
        long long v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
    return a[25];
}
""",
    ),
    "double_palindromes": dict(
        prompt="Compute the sum of all integers 1 <= n < 300000 that are palindromes both in base 10 and in base 2.",
        py_idiomatic="""def pal(s):
    return s == s[::-1]
print(sum(n for n in range(1, 300000) if pal(str(n)) and pal(bin(n)[2:])))
""",
        py_plain="""def rev(n, b):
    r = 0
    while n > 0:
        r = r * b + n % b
        n //= b
    return r
def solve():
    s = 0
    for n in range(1, 300000):
        if rev(n, 10) == n and rev(n, 2) == n:
            s += n
    return s
print(solve())
""",
        c="""static long long rev(long long n, long long b) {
    long long r = 0;
    while (n > 0) { r = r * b + n % b; n /= b; }
    return r;
}
long long solve(void) {
    long long s = 0;
    for (long long n = 1; n < 300000; n++)
        if (rev(n, 10) == n && rev(n, 2) == n) s += n;
    return s;
}
""",
    ),
    "vowels": dict(
        prompt=f'Count the vowels (a, e, i, o, u) in the string "{SENTENCE}".',
        py_idiomatic=f"""s = "{SENTENCE}"
print(sum(c in "aeiou" for c in s))
""",
        py_plain=f"""def solve():
    s = "{SENTENCE}"
    n = 0
    for c in s:
        if c == "a" or c == "e" or c == "i" or c == "o" or c == "u":
            n += 1
    return n
print(solve())
""",
        c=f"""static const char *s = "{SENTENCE}";
long long solve(void) {{
    long long n = 0;
    for (const char *p = s; *p; p++)
        if (*p == 'a' || *p == 'e' || *p == 'i' || *p == 'o' || *p == 'u') n++;
    return n;
}}
""",
    ),
    "lattice_paths": dict(
        prompt="Count the monotone lattice paths (moves right or down only) from the top-left to the bottom-right corner of a 16x16 grid of squares.",
        py_idiomatic="""n = 16
row = [1] * (n + 1)
for _ in range(n):
    for j in range(1, n + 1):
        row[j] += row[j - 1]
print(row[n])
""",
        py_plain="""def solve():
    row = [1] * 17
    for _ in range(16):
        for j in range(1, 17):
            row[j] += row[j - 1]
    return row[16]
print(solve())
""",
        c="""static long long row[17];
long long solve(void) {
    for (int j = 0; j <= 16; j++) row[j] = 1;
    for (int i = 0; i < 16; i++)
        for (int j = 1; j <= 16; j++) row[j] += row[j - 1];
    return row[16];
}
""",
    ),
}

C_MAIN = """#include <stdio.h>
{body}int main(void) {{ printf("%lld\\n", solve()); return 0; }}
"""
