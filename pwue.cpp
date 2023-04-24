#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include <immintrin.h>

using vec64_4 = uint64_t __attribute__((vector_size(32)));

constexpr uint64_t L = 0x1111111111111111;
constexpr vec64_4 vL = {L, L, L, L};

namespace vec
{
    uint64_t get(vec64_4 const &v, size_t i, size_t j)
    {
        return (v[i] >> (j << 2)) & 15;
    }

    void set(vec64_4 &v, size_t i, size_t j, uint64_t x)
    {
        v[i] &= ~(15ULL << (j << 2));
        v[i] |= x << (j << 2);
    }

    constexpr vec64_4 magic(vec64_4 x, vec64_4 y, vec64_4 L) noexcept
    {
        return (x | y) - (((x ^ y) & ~L) >> 1);
    }

    constexpr vec64_4 gamma(vec64_4 p, size_t n, size_t i) noexcept
    {
        vec64_4 q = (p >> ((i + 1) << 2)), e = (p >> (i << 2)) & 15;
        for (size_t j = 0; j < i; ++j)
            q = (q << 4) | (p & 15), p >>= 4;
        return (q - ((magic(q, ~(e * vL), vL) >> 3) & vL)) & ((1 << ((vec64_4{n, n, n, n} - 1) << 2)) - 1);
    }

    constexpr vec64_4 popcount(vec64_4 const &x) noexcept
    {
        return (vec64_4){(uint64_t)std::popcount(x[0]), (uint64_t)std::popcount(x[1]),
                         (uint64_t)std::popcount(x[2]), (uint64_t)std::popcount(x[3])};
    }

    vec64_4 mu(vec64_4 p, size_t n)
    {
        vec64_4 mu = {0, 0, 0, 0};
        vec64_4 t = {0, 0, 0, 0}, n_vec = {n, n, n, n};
        for (uint64_t j = 0; j < n; ++j)
        {
            mu = (mu * (n_vec - vec64_4{j, j, j, j})) + (p & 15) - popcount(t >> (n_vec - (p & 15)));
            t ^= 1 << (n_vec - (p & 15));
            p >>= 4;
        }
        return mu;
    }

    void next_permutation(vec64_4 &p, size_t n, size_t k)
    {
        size_t i = n - 1;
        while (i && get(p, k, i - 1) > get(p, k, i))
            --i;
        if (!i)
            return;

        size_t j = n - 1;
        while (get(p, k, j) <= get(p, k, i - 1))
            --j;

        uint64_t t = get(p, k, i - 1);
        set(p, k, i - 1, get(p, k, j));
        set(p, k, j, t);
        j = n - 1;
        while (i < j)
        {
            t = get(p, k, i);
            set(p, k, i, get(p, k, j));
            set(p, k, j, t);
            ++i, --j;
        }
    }

    void generate_permutations(vec64_4 &p, size_t n)
    {
        for (size_t i = 1; i < 4; ++i)
        {
            p[i] = p[i - 1];
            next_permutation(p, n, i);
        }
    }

    void calc_factorial_digits(vec64_4 &digits, size_t n, uint64_t i, size_t k)
    {
        for (size_t j = 1; j <= n; ++j)
        {
            set(digits, k, n - j, i % j);
            i /= j;
        }
    }

    void ith_permutation(vec64_4 &p, size_t n, uint64_t i, size_t k)
    {
        calc_factorial_digits(p, n, i, k);

        size_t const lgn = std::countl_zero<size_t>(0) - std::countl_zero(n),
                     m = 1 << lgn;
        unsigned tree[2 * m];
        for (size_t l = 0; l <= lgn; ++l)
            for (size_t j = 0; j < (1U << l); j++)
                tree[(1 << l) + j] = 1 << (lgn - l);

        for (size_t j = 0; j < n; ++j)
        {
            size_t z = 1;
            for (size_t l = 0; l < lgn; l++)
            {
                tree[z]--;
                z <<= 1;
                if (get(p, k, j) >= tree[z])
                    set(p, k, j, get(p, k, j) - tree[z++]);
            }
            tree[z] = 0;
            set(p, k, j, z - m);
        }
    }
}; // namespace vec

constexpr uint64_t factorial(uint64_t n) noexcept
{
    return n ? n * factorial(n - 1) : 1;
}

void dp(size_t n, uint8_t const *const y, uint8_t *const z, uint64_t i1, uint64_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 1);
    vec64_4 p = {0, 0, 0, 0};
    vec::ith_permutation(p, n, i1, 0);
    vec::generate_permutations(p, n);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        z[i] = z[i + 1] = z[i + 2] = z[i + 3] = n;
        z[u - i - 1] = z[u - i - 2] = z[u - i - 3] = z[u - i - 4] = n;

        for (size_t j = 0; j < n; ++j)
        {
            vec64_4 const l = vec::mu(vec::gamma(p, n, j), n - 1);

            z[i] = std::min<uint64_t>(z[i], y[l[0]] + 1);
            z[i + 1] = std::min<uint64_t>(z[i + 1], y[l[1]] + 1);
            z[i + 2] = std::min<uint64_t>(z[i + 2], y[l[2]] + 1);
            z[i + 3] = std::min<uint64_t>(z[i + 3], y[l[3]] + 1);

            z[u - i - 1] = std::min<uint64_t>(z[u - i - 1], y[v - l[0] - 1] + 1);
            z[u - i - 2] = std::min<uint64_t>(z[u - i - 2], y[v - l[1] - 1] + 1);
            z[u - i - 3] = std::min<uint64_t>(z[u - i - 3], y[v - l[2] - 1] + 1);
            z[u - i - 4] = std::min<uint64_t>(z[u - i - 4], y[v - l[3] - 1] + 1);
        }

        if (i + 4 < i2)
        {
            p[0] = p[3];
            vec::next_permutation(p, n, 0);
            vec::generate_permutations(p, n);
        }
    }
}

// Gehe 1 Ebene herunter.
std::pair<uint64_t, uint64_t> bf1(
    size_t n, uint8_t const *const y, size_t i1, size_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 1);
    std::pair<uint64_t, uint64_t> result = {0, 0};
    vec64_4 p = {0, 0, 0, 0};
    vec::ith_permutation(p, n, i1, 0);
    vec::generate_permutations(p, n);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        vec64_4 a1 = {n, n, n, n}, a2 = {n, n, n, n};

        for (size_t j = 0; j < n; ++j)
        {
            vec64_4 const l = vec::mu(vec::gamma(p, n, j), n - 1);

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

        if (i + 4 < i2)
        {
            p[0] = p[3];
            vec::next_permutation(p, n, 0);
            vec::generate_permutations(p, n);
        }
    }

    return result;
}

// Gehe 2 Ebenen herunter.
std::pair<uint64_t, uint64_t> bf2(
    size_t n, uint8_t const *const y, size_t i1, size_t i2)
{
    uint64_t const u = factorial(n), v = factorial(n - 2);
    std::pair<uint64_t, uint64_t> result = {0, 0};
    vec64_4 p = {0, 0, 0, 0};
    vec::ith_permutation(p, n, i1, 0);
    vec::generate_permutations(p, n);

    for (uint64_t i = i1; i < i2; i += 4)
    {
        vec64_4 a1 = {n, n, n, n}, a2 = {n, n, n, n};

        for (size_t j = 0; j < n; ++j)
        {
            vec64_4 q = vec::gamma(p, n, j);
            for (size_t k = 0; k < n - 1; ++k)
            {
                vec64_4 const l = vec::mu(vec::gamma(q, n - 1, k), n - 2);

                a1[0] = std::min<uint64_t>(a1[0], y[l[0]] + 2);
                a1[1] = std::min<uint64_t>(a1[1], y[l[1]] + 2);
                a1[2] = std::min<uint64_t>(a1[2], y[l[2]] + 2);
                a1[3] = std::min<uint64_t>(a1[3], y[l[3]] + 2);

                a2[0] = std::min<uint64_t>(a2[0], y[v - l[0] - 1] + 2);
                a2[1] = std::min<uint64_t>(a2[1], y[v - l[1] - 1] + 2);
                a2[2] = std::min<uint64_t>(a2[2], y[v - l[2] - 1] + 2);
                a2[3] = std::min<uint64_t>(a2[3], y[v - l[3] - 1] + 2);

                if (a1[0] < 8 && a1[1] < 8 && a1[2] < 8 && a1[3] < 8 &&
                    a2[0] < 8 && a2[1] < 8 && a2[2] < 8 && a2[3] < 8)
                    goto next_permutation_block;
            }
        }

    next_permutation_block:
        for (size_t j = 0; j < 4; ++j)
        {
            if (a1[j] > result.first)
                result = {a1[j], i + j};
            if (a2[j] > result.first)
                result = {a2[j], u - i - 1 - j};
        }

        if (i + 4 < i2)
        {
            p[0] = p[3];
            vec::next_permutation(p, n, 0);
            vec::generate_permutations(p, n);
        }
    }

    return result;
}

std::pair<uint64_t, uint64_t> get_mu_interval(size_t n)
{
    std::cout << "Intervall an zu berechnenden Permutationen einschränken? [y/n] ";
    char c;
    std::cin >> c;
    if (c == 'n')
        return {0, factorial(n)};
    uint64_t I, J;
    std::cout << "Die Länge des Intervalls muss ein Vielfaches von "
              << 8 * std::thread::hardware_concurrency() << " sein.\n"
              << "Anfang (inklusiv): ";
    std::cin >> I;
    std::cout << "Ende (exklusiv): ";
    std::cin >> J;
    return {I, J};
}

int main()
{
    std::cout << "n: ";
    size_t n;
    std::cin >> n;
    if (n < 4)
    {
        std::cout << "Funktioniert nur für n >= 4...\n";
        return 0;
    }

    auto const [I, J] = get_mu_interval(n);
    auto const start_time = std::chrono::system_clock::now();

    uint8_t *a = (uint8_t *)malloc(factorial(std::min<size_t>(n - 1, 13))),
            *b = (uint8_t *)malloc(factorial(std::min<size_t>(n - 1, 12)));
    a[0] = 0, a[1] = 1, a[2] = 1, a[3] = 2, a[4] = 1, a[5] = 1;
    size_t const num_threads = std::thread::hardware_concurrency();

    for (size_t k = 4; k <= std::min<size_t>(n - 1, 13); ++k)
    {
        std::vector<std::thread> threads;
        if (!(factorial(k) % (num_threads * 8)))
        {
            for (size_t t = 0; t < num_threads; t++)
                threads.emplace_back(dp, k, (uint8_t *)((k & 1) ? b : a),
                                     (uint8_t *)((k & 1) ? a : b),
                                     (factorial(k) / 2) * t / num_threads,
                                     (factorial(k) / 2) * (t + 1) / num_threads);
        }
        else
        {
            dp(k, (uint8_t *)((k & 1) ? b : a), (uint8_t *)((k & 1) ? a : b),
               0, factorial(k) / 2);
        }
        for (std::thread &t : threads)
            t.join();
        ((k & 1) ? a : b)[0] = 0;
    }

    std::pair<uint64_t, uint64_t> result = {0, 0};
    std::vector<std::future<std::pair<uint64_t, uint64_t>>> fut;

    if (!(factorial(n) % (num_threads * 8)))
    {
        for (size_t t = 0; t < num_threads; ++t)
            fut.emplace_back(async(n <= 14 ? bf1 : bf2, n,
                                   (uint8_t *)((n & 1) && n != 15 ? b : a),
                                   I + ((J - I) / 2) * t / num_threads,
                                   I + ((J - I) / 2) * (t + 1) / num_threads));
    }
    else
    {
        fut.emplace_back(async(n <= 14 ? bf1 : bf2, n,
                               (uint8_t *)((n & 1) && n != 15 ? b : a), I, I + (J - I) / 2));
    }

    for (auto &f : fut) // Finde das maximale A(p) aller Threads.
        result = max(result, f.get());

    free(a);
    free(b);
    vec64_4 p = {0, 0, 0, 0};
    vec::ith_permutation(p, n, result.second, 0);
    std::cout << "Laufzeit: "
              << std::chrono::duration_cast<std::chrono::milliseconds>((std::chrono::system_clock::now() - start_time)).count()
              << " ms\ngrößtes gefundenes A(p): " << result.first
              << "\nBeispiel: ";
    for (size_t i = 0; i < n; ++i)
        std::cout << vec::get(p, 0, i) + 1 << ' ';
    std::cout << '\n';
}