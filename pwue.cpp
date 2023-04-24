#include <algorithm>
#include <bit>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include <immintrin.h>

constexpr size_t N = 15;

using vec64_4 = uint64_t __attribute__((vector_size(32)));

constexpr uint64_t factorial(uint64_t n) noexcept
{
    return n ? n * factorial(n - 1) : 1;
}

void gamma(size_t n, vec64_4 *p, vec64_4 *q, size_t i)
{
    for (size_t j = 0; j < i; j++)
        q[j] = p[i - j - 1] + (p[i - j - 1] > p[i]);
    for (size_t j = i; j < n - 1; j++)
        q[j] = p[j + 1] + (p[j + 1] > p[i]);
}

vec64_4 vec_popcount(vec64_4 const &x)
{
    return (vec64_4){(uint64_t)std::popcount(x[0]), (uint64_t)std::popcount(x[1]),
                     (uint64_t)std::popcount(x[2]), (uint64_t)std::popcount(x[3])};
}

vec64_4 linear_mu(size_t n, vec64_4 *p)
{
    vec64_4 mu = {0, 0, 0, 0};
    vec64_4 t = {0, 0, 0, 0}, n_vec = {n, n, n, n};
    for (uint64_t j = 0; j < n; ++j)
    {
        mu = (mu * (n_vec - (vec64_4){j, j, j, j})) - vec_popcount(t >> (n_vec - p[j]));
        t ^= 1 << (n_vec - p[j]);
    }
    return mu;
}

vec64_4 linear_mu_gamma(size_t n, vec64_4 *p, size_t i)
{
    vec64_4 mu = {0, 0, 0, 0};
    vec64_4 t = {0, 0, 0, 0}, n_vec = {n, n, n, n};
    for (uint64_t j = 0; j < i; ++j)
    {
        vec64_4 const x = p[i - j - 1] + (p[i - j - 1] > p[i]);
        mu = (mu * (n_vec - (vec64_4){j, j, j, j} - 1)) + x - vec_popcount(t >> (n_vec - x));
        t ^= 1 << (n_vec - x);
    }
    for (uint64_t j = i; j < n - 1; ++j)
    {
        vec64_4 const x = p[j + 1] + (p[j + 1] > p[i]);
        mu = (mu * (n_vec - (vec64_4){j, j, j, j} - 1)) + x - vec_popcount(t >> (n_vec - x));
        t ^= 1 << (n_vec - x);
    }
    return mu;
}

void vec_calc_factorial_digits(size_t n, uint64_t i, vec64_4 *digits, size_t k)
{
    for (size_t j = 1; j <= n; j++)
    {
        digits[n - j][k] = i % j;
        i /= j;
    }
}

// Writes the lexicographically i-th permutation of size n at position k in p.
void vec_ith_permutation(size_t n, uint64_t i, vec64_4 *p, size_t k)
{
    vec_calc_factorial_digits(n, i, p, k);

    size_t const lgn = std::countl_zero<size_t>(0) - std::countl_zero(n),
                 m = 1 << lgn;
    unsigned tree[2 * m];
    for (size_t l = 0; l <= lgn; l++)
        for (size_t j = 0; j < (1U << l); j++)
            tree[(1 << l) + j] = 1 << (lgn - l);

    for (size_t j = 0; j < n; j++)
    {
        size_t z = 1;
        for (size_t l = 0; l < lgn; l++)
        {
            tree[z]--;
            z <<= 1;
            if (p[j][k] >= tree[z])
                p[j][k] -= tree[z++];
        }
        tree[z] = 0;
        p[j][k] = z - m;
    }
}

void vec_next_permutation(size_t n, vec64_4 *p, size_t k)
{
    size_t i = n - 1;
    while (i && p[i - 1][k] > p[i][k])
        i--;
    if (!i)
        return;

    size_t j = n - 1;
    while (p[j][k] <= p[i - 1][k])
        j--;

    std::swap(p[i - 1][k], p[j][k]);
    j = n - 1;
    while (i < j)
        std::swap(p[i++][k], p[j--][k]);
}

void vec_generate_permutations(size_t n, vec64_4 *p)
{
    for (size_t i = 1; i < 4; ++i)
    {
        for (size_t j = 0; j < n; ++j)
            p[j][i] = p[j][i - 1];
        vec_next_permutation(n, p, i);
    }
}

void dp(size_t n, uint8_t const *const y, uint8_t *const z, uint64_t i1, uint64_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 1);
    vec64_4 p[N];
    vec_ith_permutation(n, i1, p, 0);
    vec_generate_permutations(n, p);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        z[i] = z[i + 1] = z[i + 2] = z[i + 3] = n;
        z[u - i - 1] = z[u - i - 2] = z[u - i - 3] = z[u - i - 4] = n;

        for (size_t j = 0; j < n; ++j)
        {
            vec64_4 const l = linear_mu_gamma(n, p, j);

            z[i] = std::min<uint64_t>(z[i], y[l[0]] + 1);
            z[i + 1] = std::min<uint64_t>(z[i + 1], y[l[1]] + 1);
            z[i + 2] = std::min<uint64_t>(z[i + 2], y[l[2]] + 1);
            z[i + 3] = std::min<uint64_t>(z[i + 3], y[l[3]] + 1);

            z[u - i - 1] = std::min<uint64_t>(z[u - i - 1], y[v - l[0] - 1] + 1);
            z[u - i - 2] = std::min<uint64_t>(z[u - i - 2], y[v - l[1] - 1] + 1);
            z[u - i - 3] = std::min<uint64_t>(z[u - i - 3], y[v - l[2] - 1] + 1);
            z[u - i - 4] = std::min<uint64_t>(z[u - i - 4], y[v - l[3] - 1] + 1);
        }

        for (size_t i = 0; i < n; ++i)
            p[i][0] = p[i][3];
        vec_next_permutation(n, p, 0);
        vec_generate_permutations(n, p);
    }
}

// Gehe 1 Ebene herunter.
std::pair<uint64_t, uint64_t> bf1(
    size_t n, uint8_t const *const y, size_t i1, size_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 1);
    std::pair<uint64_t, uint64_t> result = {0, 0};
    vec64_4 p[N];
    vec_ith_permutation(n, i1, p, 0);
    vec_generate_permutations(n, p);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        vec64_4 a1 = {n, n, n, n}, a2 = {n, n, n, n};

        for (size_t j = 0; j < n; ++j)
        {
            vec64_4 const l = linear_mu_gamma(n, p, j);

            a1[0] = std::min<uint64_t>(a1[0], y[l[0]] + 1);
            a1[1] = std::min<uint64_t>(a1[1], y[l[1]] + 1);
            a1[2] = std::min<uint64_t>(a1[2], y[l[2]] + 1);
            a1[3] = std::min<uint64_t>(a1[3], y[l[3]] + 1);

            a2[0] = std::min<uint64_t>(a2[0], y[v - l[0] - 1] + 1);
            a2[1] = std::min<uint64_t>(a2[1], y[v - l[1] - 1] + 1);
            a2[2] = std::min<uint64_t>(a2[2], y[v - l[2] - 1] + 1);
            a2[3] = std::min<uint64_t>(a2[3], y[v - l[3] - 1] + 1);
        }

        for (size_t j = 0; j < 4; ++j)
        {
            if (a1[j] > result.first)
                result = {a1[j], i + j};
            if (a2[j] > result.first)
                result = {a2[j], u - i - 1 - j};
        }

        for (size_t i = 0; i < n; ++i)
            p[i][0] = p[i][3];
        vec_next_permutation(n, p, 0);
        vec_generate_permutations(n, p);
    }

    return result;
}

// Gehe 2 Ebenen herunter.
std::pair<uint64_t, uint64_t> bf2(
    size_t n, uint8_t const *const y, size_t i1, size_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 2);
    std::pair<uint64_t, uint64_t> result = {0, 0};
    vec64_4 p[N], q[N];
    vec_ith_permutation(n, i1, p, 0);
    vec_generate_permutations(n, p);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        vec64_4 a1 = {n, n, n, n}, a2 = {n, n, n, n};

        for (size_t j = 0; j < n; ++j)
        {
            gamma(n, p, q, j);
            for (size_t k = 0; k < n - 1; ++k)
            {
                vec64_4 const l = linear_mu_gamma(n - 1, q, k);

                a1[0] = std::min<uint64_t>(a1[0], y[l[0]] + 2);
                a1[1] = std::min<uint64_t>(a1[1], y[l[1]] + 2);
                a1[2] = std::min<uint64_t>(a1[2], y[l[2]] + 2);
                a1[3] = std::min<uint64_t>(a1[3], y[l[3]] + 2);

                a2[0] = std::min<uint64_t>(a2[0], y[v - l[0] - 1] + 2);
                a2[1] = std::min<uint64_t>(a2[1], y[v - l[1] - 1] + 2);
                a2[2] = std::min<uint64_t>(a2[2], y[v - l[2] - 1] + 2);
                a2[3] = std::min<uint64_t>(a2[3], y[v - l[3] - 1] + 2);
            }
        }

        for (size_t j = 0; j < 4; ++j)
        {
            if (a1[j] > result.first)
                result = {a1[j], i + j};
            if (a2[j] > result.first)
                result = {a2[j], u - i - 1 - j};
        }

        for (size_t i = 0; i < n; ++i)
            p[i][0] = p[i][3];
        vec_next_permutation(n, p, 0);
        vec_generate_permutations(n, p);
    }

    return result;
}

int main()
{
    size_t n;
    std::cin >> n;
    if (n < 6)
    {
        std::cout << "Funktioniert nur für n >= 6...\n";
        return 0;
    }

    uint8_t *a = (uint8_t *)malloc(factorial(12)),
            *b = (uint8_t *)malloc(factorial(12));
    a[0] = 0, a[1] = 1, a[2] = 1, a[3] = 2, a[4] = 1, a[5] = 1;
    size_t const num_threads = std::thread::hardware_concurrency();

    for (size_t k = 4; k <= std::min<size_t>(n - 1, 13); ++k)
    {
        std::vector<std::thread> threads;
        if (!(factorial(k) & 15))
        {
            for (size_t t = 0; t < num_threads; t++)
                threads.emplace_back(dp, k, (uint8_t *)((k & 1) ? b : a),
                                     (uint8_t *)((k & 1) ? a : b),
                                     (factorial(k) / 2) * t / num_threads,
                                     (factorial(k) / 2) * (t + 1) / num_threads);
        }
        else
        {
            threads.emplace_back(dp, k, (uint8_t *)((k & 1) ? b : a),
                                 (uint8_t *)((k & 1) ? a : b),
                                 0, factorial(k) / 2);
        }
        for (std::thread &t : threads)
            t.join();
        ((k & 1) ? a : b)[0] = 0;
    }

    std::pair<uint64_t, uint64_t> result = {0, 0};
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> fut;

    for (size_t t = 0; t < num_threads; ++t)
        fut.emplace_back(async(n <= 14 ? bf1 : bf2, n,
                               (uint8_t *)((n & 1) && n != 15 ? b : a),
                               (factorial(n) / 2) * t / num_threads,
                               (factorial(n) / 2) * (t + 1) / num_threads));

    for (auto &f : fut) // Finde das maximale A(p) aller Threads.
        result = max(result, f.get());

    free(a);
    free(b);
    vec64_4 p[N];
    vec_ith_permutation(n, result.second, p, 0);
    std::cout << "P(" << n << ") = " << result.first
              << ", Beispiel mit A(p) = P(" << n << ") : ";
    for (size_t i = 0; i < n; ++i)
        std::cout << p[i][0] + 1 << ' ';
    std::cout << '\n';
}