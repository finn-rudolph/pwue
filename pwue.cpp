#include <algorithm>
#include <bit>
#include <cstdint>
#include <future>
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <immintrin.h>

constexpr size_t N = 18;

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

std::pair<uint64_t, uint64_t> bf5(
    uint8_t const *const y, size_t i1, size_t i2, std::shared_mutex *mut)
{
    uint64_t const u = factorial(18), v = factorial(13);
    std::pair<uint64_t, uint64_t> result = {0, 0};
    vec64_4 p[N], q1[N], q2[N], q3[N], q4[N];
    vec_ith_permutation(18, i1, p, 0);
    vec_generate_permutations(18, p);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        vec64_4 a1 = {18, 18, 18, 18}, a2 = {18, 18, 18, 18};

        for (size_t j1 = 0; j1 < 18; ++j1)
        {
            gamma(18, p, q1, j1);
            for (size_t j2 = 0; j2 < 17; ++j2)
            {
                gamma(17, q1, q2, j2);
                for (size_t j3 = 0; j3 < 16; ++j3)
                {
                    gamma(16, q2, q3, j3);
                    for (size_t j4 = 0; j4 < 15; ++j4)
                    {
                        gamma(15, q3, q4, j4);
                        for (size_t j5 = 0; j5 < 14; ++j5)
                        {
                            vec64_4 const l = linear_mu_gamma(14, q4, j5);

                            a1[0] = std::min<uint64_t>(a1[0], y[l[0]] + 2);
                            a1[1] = std::min<uint64_t>(a1[1], y[l[1]] + 2);
                            a1[2] = std::min<uint64_t>(a1[2], y[l[2]] + 2);
                            a1[3] = std::min<uint64_t>(a1[3], y[l[3]] + 2);

                            a2[0] = std::min<uint64_t>(a2[0], y[v - l[0] - 1] + 2);
                            a2[1] = std::min<uint64_t>(a2[1], y[v - l[1] - 1] + 2);
                            a2[2] = std::min<uint64_t>(a2[2], y[v - l[2] - 1] + 2);
                            a2[3] = std::min<uint64_t>(a2[3], y[v - l[3] - 1] + 2);

                            if (a1[0] < 11 && a1[1] < 11 && a1[2] < 11 && a1[3] < 11 &&
                                a2[0] < 11 && a2[1] < 11 && a2[2] < 11 && a2[3] < 11)
                                goto next_permutation_block;
                        }
                    }
                }
            }
        }

        for (size_t j = 0; j < 4; ++j)
        {
            if (a1[j] == 11)
            {
                std::unique_lock lck(*mut);
                for (size_t i = 0; i < 18; ++i)
                    std::cout << p[i][j] + 1 << ' ';
                std::cout << std::endl;
            }
            if (a2[j] == 11)
            {
                vec_ith_permutation(18, u - i - 1 - j, q1, 0);
                std::unique_lock lck(*mut);
                for (size_t i = 0; i < 18; ++i)
                    std::cout << q1[i][0] + 1 << ' ';
                std::cout << std::endl;
            }
            if (a1[j] > result.first)
                result = {a1[j], i + j};
            if (a2[j] > result.first)
                result = {a2[j], u - i - 1 - j};
        }

    next_permutation_block:
        for (size_t i = 0; i < 18; ++i)
            p[i][0] = p[i][3];
        vec_next_permutation(18, p, 0);
        vec_generate_permutations(18, p);
    }

    return result;
}

std::pair<uint64_t, uint64_t> get_mu_interval(size_t n)
{
    std::cout << "Intervall an zu berechnenden Permutationen einschränken? [y/n] ";
    char c;
    std::cin >> c;
    if (c == 'n')
        return {0, factorial(n) / 2};
    uint64_t mu_begin, mu_end;
    std::cout << "Das Intervall muss Teil von [0, n! / 2) = [0, " << factorial(n) / 2 << ") sein.\n"
              << "Die Länge des Intervalls muss ein Vielfaches von "
              << 8 * std::thread::hardware_concurrency() << " sein.\n"
              << "Anfang (inklusiv): ";
    std::cin >> mu_begin;
    std::cout << "Ende (exklusiv): ";
    std::cin >> mu_end;
    return {mu_begin, mu_end};
}

int main()
{
    size_t n;
    std::cin >> n;
    if (n < 4)
    {
        std::cout << "Funktioniert nur für n >= 4...\n";
        return 0;
    }

    auto const [mu_begin, mu_end] = get_mu_interval(n);
    auto const start_time = std::chrono::system_clock::now();

    uint8_t *a = (uint8_t *)malloc(factorial(13)),
            *b = (uint8_t *)malloc(factorial(12));
    a[0] = 0, a[1] = 1, a[2] = 1, a[3] = 2, a[4] = 1, a[5] = 1;
    size_t const num_threads = std::thread::hardware_concurrency();

    for (size_t k = 4; k <= 13; ++k)
    {
        std::vector<std::thread> threads;
        if (!(factorial(k) & 15))
        {
            for (size_t t = 0; t < num_threads; t++)
                threads.emplace_back(dp, k, a, b,
                                     (factorial(k) / 2) * t / num_threads,
                                     (factorial(k) / 2) * (t + 1) / num_threads);
        }
        else
        {
            threads.emplace_back(dp, k, a, b, 0, factorial(k) / 2);
        }
        for (std::thread &t : threads)
            t.join();
        std::swap(a, b);
    }

    std::pair<uint64_t, uint64_t> result = {0, 0};
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> fut;
    std::shared_mutex mut;

    for (size_t t = 0; t < num_threads; ++t)
        fut.emplace_back(async(bf5, a,
                               mu_begin + (mu_end - mu_begin) * t / num_threads,
                               mu_begin + (mu_end - mu_begin) * (t + 1) / num_threads, &mut));

    for (auto &f : fut) // Finde das maximale A(p) aller Threads.
        result = max(result, f.get());

    free(a);
    free(b);
    vec64_4 p[N];
    vec_ith_permutation(n, result.second, p, 0);
    std::cout << "Laufzeit: "
              << std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::system_clock::now() - start_time)).count()
              << "P(" << n << ") = " << result.first
              << ", Beispiel mit A(p) = P(" << n << ") : ";
    for (size_t i = 0; i < n; ++i)
        std::cout << p[i][0] + 1 << ' ';
    std::cout << '\n';
}