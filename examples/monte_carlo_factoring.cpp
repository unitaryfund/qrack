////////////////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// "A quantum-inspired Monte Carlo integer factoring algorithm"
//
// This example demonstrates a (Shor's-like) "quantum-inspired" algorithm for integer factoring.
// This approach is similar to Shor's algorithm, except with a uniformly random output from the
// quantum period-finding subroutine. Therefore, we don't need quantum computer simulation for
// this algorithm at all!
//
// (This file was heavily adapted from
// https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/shor.py,
// with thanks to ProjectQ!)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <chrono>
#include <cmath>
#include <iomanip> // For setw
#include <iostream> // For cout
#include <random>
#include <stdlib.h>
#include <time.h>

#include <atomic>
#include <future>
#include <mutex>

// Turn this off, if you're not factoring a semi-prime number with equal-bit-width factors.
#define IS_SEMI_PRIME 1

#define ONE_BCI ((bitCapInt)1UL)
#define bitsInByte 8
// Change QBCAPPOW, if you need more than 2^6 bits of factorized integer, within Boost and system limits.
// (2^7, only, needs custom std::cout << operator implementation.)
#define QBCAPPOW 6U

#if QBCAPPOW < 8U
#define bitLenInt uint8_t
#elif QBCAPPOW < 16U
#define bitLenInt uint16_t
#elif QBCAPPOW < 32U
#define bitLenInt uint32_t
#else
#define bitLenInt uint64_t
#endif

#if QBCAPPOW < 6U
#define bitsInCap 32
#define bitCapInt uint32_t
#elif QBCAPPOW < 7U
#define bitsInCap 64
#define bitCapInt uint64_t
#elif QBCAPPOW < 8U
#define bitsInCap 128
#ifdef BOOST_AVAILABLE
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt boost::multiprecision::uint128_t
#else
#define bitCapInt __uint128_t
#endif
#else
#define bitsInCap (8U * (((bitLenInt)1U) << QBCAPPOW))
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1 << QBCAPPOW, 1 << QBCAPPOW,                 \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#endif

// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x) { return (x && !(x & (x - ONE_BCI))); }

inline bitLenInt log2(const bitCapInt& n)
{
#if __GNUC__
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#elif QBCAPPOW < 7
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#else
    const size_t bitsInWord = bitsInByte * sizeof(unsigned long long);
    int nZeroes = __builtin_clzll((unsigned long long)(n & 0xFFFFFFFFFFFFFFFF));
    bitLenInt pow = 0U;
    while (nZeroes == bitsInWord) {
        pow += bitsInWord;
        n >>= bitsInWord;
        nZeroes = __builtin_clzll((unsigned long long)(n & 0xFFFFFFFFFFFFFFFF));
    }
    return pow + (bitsInWord - nZeroes - 1U);
#endif
#else
    bitLenInt pow = 0;
    bitCapInt p = n >> 1U;
    while (p) {
        p >>= 1U;
        pow++;
    }
    return pow;
#endif
}

// Source:
// https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int#answer-101613
inline bitCapInt uipow(const bitCapInt& base, const bitCapInt& exp)
{
    bitCapInt result = 1U;
    bitCapInt b = base;
    bitCapInt e = exp;
    for (;;) {
        if (b & 1U) {
            result *= b;
        }
        e >>= 1U;
        if (!e) {
            break;
        }
        b *= b;
    }

    return result;
}

// It's fine if this is not exact for the whole bitCapInt domain, so long as it is <= the exact result.
inline bitLenInt intLog(const bitCapInt& base, const bitCapInt& arg)
{
    bitLenInt result = 0U;
    for (bitCapInt x = arg; x >= base; x /= base) {
        result++;
    }
    return result;
}

bitCapInt gcd(const bitCapInt& n1, const bitCapInt& n2)
{
    if (n2) {
        return gcd(n2, n1 % n2);
    }
    return n1;
}

int main()
{
    typedef std::uniform_int_distribution<uint64_t> rand_dist;

    bitCapInt toFactor;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

    const double clockFactor = 1.0 / 1000.0; // Report in ms
    auto iterClock = std::chrono::high_resolution_clock::now();

    const bitLenInt qubitCount = log2(toFactor) + (isPowerOfTwo(toFactor) ? 0U : 1U);
    const bitCapInt qubitPower = ONE_BCI << qubitCount;
    std::cout << "Bits to factor: " << (int)qubitCount << std::endl;

#if IS_SEMI_PRIME
    const bitCapInt baseMin = 1U << ((qubitCount - 1U) / 2 - 1U);
    const bitCapInt baseMax = baseMin << 1U;
#else
    const bitCapInt baseMin = 2U;
    const bitCapInt baseMax = (toFactor - 1U);
#endif

    std::vector<rand_dist> toFactorDist;
#if QBCAPPOW > 6U
    const bitLenInt wordSize = 64U;
    const bitCapInt wordMask = 0xFFFFFFFFFFFFFFFF;
    bitCapInt distPart = baseMax - baseMin;
    while (distPart) {
        toFactorDist.push_back(rand_dist(0U, (uint64_t)(distPart & wordMask)));
        distPart >>= wordSize;
    }
    std::reverse(toFactorDist.begin(), toFactorDist.end());
#else
    toFactorDist.push_back(rand_dist(baseMin, baseMax));
#endif

    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());

    const unsigned threads = std::thread::hardware_concurrency();
    std::atomic<bool> isFinished;
    isFinished = false;

    std::vector<std::future<void>> futures(threads);
    for (unsigned cpu = 0U; cpu < threads; cpu++) {
        futures[cpu] = std::async(std::launch::async, [&] {
            const size_t BATCH_SIZE = 1U << 9U;

            for (;;) {
                for (size_t batchItem = 0U; batchItem < BATCH_SIZE; batchItem++) {
                    // Choose a base at random, >1 and <toFactor.
                    bitCapInt base = toFactorDist[0](rand_gen);
#if QBCAPPOW > 6U
                    for (size_t i = 1U; i < toFactorDist.size(); i++) {
                        base <<= wordSize;
                        base |= toFactorDist[i](rand_gen);
                    }
                    base += baseMin;
#endif

#if IS_SEMI_PRIME
                    // We assume there's no particular downside to choosing only odd bases,
                    // which might be more likely to immediately yield a prime.
                    base = (base << 1U) | 1U;
#endif

                    const bitCapInt testFactor = gcd(toFactor, base);
                    if (testFactor != 1) {
                        std::cout << "Chose non-relative prime: " << testFactor << " * " << (toFactor / testFactor)
                                  << std::endl;
                        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - iterClock);
                        std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;
                        std::cout << "(Waiting to join other threads...)" << std::endl;
                        return;
                    }

                    // This would be where we perform the quantum period finding algorithm.
                    // However, we don't have a quantum computer!
                    // Instead, we "throw dice" for a guess to the output of the quantum subroutine.
                    // This guess will usually be wrong, at least for semi-prime inputs.
                    // If we try many times, though, this can be a practically valuable factoring method.

                    // y is meant to be close to some number c * qubitPower / r, where r is the period.
                    // c is a positive integer or 0, and we don't want the 0 case.
                    // y is truncated by the number of qubits in the register, at most.
                    // The maximum value of c before truncation is no higher than r.

                    // The period of ((base ^ x) MOD toFactor) can't be smaller than log_base(toFactor).
                    // (Also, toFactor is definitely NOT an exact multiple of base.)
                    const bitCapInt minR = (bitCapInt)intLog(base, toFactor) + 1U;
                    // It can be shown that the period of this modular exponentiation can be no higher than 1
                    // less than the modulus, as in https://www2.math.upenn.edu/~mlazar/math170/notes06-3.pdf.
                    const bitCapInt maxR = toFactor - 1U;

                    // c is basically a harmonic degeneracy factor, and there might be no value in testing
                    // any case except c = 1, without loss of generality.

                    // This sets a nonuniform distribution on our y values to test.
                    // y values are close to qubitPower / rGuess, and we midpoint round.

                    // However, results are better with uniformity over r, rather than y.

                    // So, we guess r, between minR and maxR.
#if QBCAPPOW > 6U
                    bitCapInt rPart = maxR - minR;
                    bitCapInt r = 0U;
                    while (rPart) {
                        rand_dist rDist(0U, (uint64_t)(rPart & wordMask));
                        rPart >>= wordSize;
                        r <<= wordSize;
                        r |= rDist(rand_gen);
                    }
                    r += minR;
#else
                    rand_dist rDist(minR, maxR);
                    bitCapInt r = rDist(rand_gen);
#endif

                    // Since our output is r rather than y, we can skip the continued fractions step.

                    // Try to determine the factors
                    if (r & 1U) {
                        r <<= 1U;
                    }
                    const bitCapInt p = r >> 1U;
                    const bitCapInt apowrhalf = uipow(base, p) % toFactor;
                    bitCapInt f1 = (bitCapInt)gcd(apowrhalf + 1U, toFactor);
                    bitCapInt f2 = (bitCapInt)gcd(apowrhalf - 1U, toFactor);
                    bitCapInt fmul = f1 * f2;
                    while ((fmul != toFactor) && (fmul > 1U) && ((toFactor / fmul) * fmul == toFactor)) {
                        fmul = f1;
                        f1 = fmul * f2;
                        f2 = toFactor / (fmul * f2);
                        fmul = f1 * f2;
                    }
                    if ((fmul == toFactor) && (f1 > 1U) && (f2 > 1U)) {
                        std::cout << "Success: Found " << f1 << " * " << f2 << " = " << toFactor << std::endl;
                        auto tClock = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - iterClock);
                        std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;
                        std::cout << "(Waiting to join other threads...)" << std::endl;
                        isFinished = true;
                        return;
                    } // else {
                      // std::cout << "Failure: Found " << res1 << " and " << res2 << std::endl;
                    // }
                }

                // Check if finished, between batches.
                if (isFinished) {
                    break;
                }
            }
        });
    };

    for (unsigned cpu = 0U; cpu < threads; cpu++) {
        futures[cpu].get();
    }

    return 0;
}
