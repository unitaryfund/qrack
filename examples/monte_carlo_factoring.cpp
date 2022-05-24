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
#define IS_RSA_SEMI_PRIME 1
// Turn this off, if you don't want to coordinate across multiple (quasi-independent) nodes.
#define IS_DISTRIBUTED 1
// The maximum number of bits in Boost big integers is 2^QBCAPPOW.
// (2^7, only, needs custom std::cout << operator implementation.)
#define QBCAPPOW 8U

#define ONE_BCI ((bitCapInt)1UL)
#define bitsInByte 8U

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
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1ULL << QBCAPPOW, 1ULL << QBCAPPOW,           \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#endif

// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x) { return (x && !(x & (x - ONE_BCI))); }

inline bitLenInt log2(const bitCapInt& n)
{
#if __GNUC__ && QBCAPPOW < 7
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
    bitLenInt pow = 0U;
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

// Adapted from Gaurav Ahirwar's suggestion on https://www.geeksforgeeks.org/square-root-of-an-integer/
bitCapInt floorSqrt(const bitCapInt& x)
{
    // Base cases
    if ((x == 0) || (x == 1)) {
        return x;
    }

    // Binary search for floor(sqrt(x))
    bitCapInt start = 1U, end = x >> 1U, ans;
    while (start <= end) {
        bitCapInt mid = (start + end) >> 1U;

        // If x is a perfect square
        bitCapInt sqr = mid * mid;
        if (sqr == x) {
            return mid;
        }

        if (sqr < x) {
            // Since we need floor, we update answer when mid*mid is smaller than x, and move closer to sqrt(x).
            start = mid + 1U;
            ans = mid;
        } else {
            // If mid*mid is greater than x
            end = mid - 1U;
        }
    }
    return ans;
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
    bitCapInt nodeCount = 1U;
    bitCapInt nodeId = 0U;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

    auto iterClock = std::chrono::high_resolution_clock::now();

    const bitLenInt qubitCount = log2(toFactor) + (isPowerOfTwo(toFactor) ? 0U : 1U);
    // const bitCapInt qubitPower = ONE_BCI << qubitCount;
    std::cout << "Bits to factor: " << (int)qubitCount << std::endl;

#if IS_DISTRIBUTED
    std::cout << "You can split this work across nodes, without networking!" << std::endl;
    do {
        std::cout << "Number of nodes (>=1): ";
        std::cin >> nodeCount;
        if (!nodeCount) {
            std::cout << "Invalid node count choice!" << std::endl;
        }
    } while (!nodeCount);
    if (nodeCount > 1U) {
        do {
            std::cout << "Which node is this? (0-" << (int)(nodeCount - 1U) << "):";
            std::cin >> nodeId;
            if (nodeId >= nodeCount) {
                std::cout << "Invalid node ID choice!" << std::endl;
            }
        } while (nodeId >= nodeCount);
    }
#endif

    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());

    const unsigned cpuCount = std::thread::hardware_concurrency();
    std::atomic<bool> isFinished;
    isFinished = false;

#if IS_RSA_SEMI_PRIME
    std::map<bitLenInt, const std::vector<bitCapInt>> primeDict = { { 16U, { 32771U, 32779U, 65519U, 65521U } },
        { 28U, { 134217757U, 134217773U, 268435367U, 268435399U } },
        { 32U, { 2147483659U, 2147483693U, 4294967279U, 4294967291U } },
        { 64U, { 9223372036854775837U, 9223372036854775907U, 1844674407370955137U, 1844674407370955143U } } };

    // If n is semiprime, \phi(n) = (p - 1) * (q - 1), where "p" and "q" are prime.
    // The minimum value of this formula, for our input, without consideration of actual
    // primes in the interval, is as follows:
    // (See https://www.mobilefish.com/services/rsa_key_generation/rsa_key_generation.php)
    const bitLenInt primeBits = (qubitCount + 1U) >> 1U;
    const bitCapInt fullMin = ONE_BCI << (primeBits - 1U);
    const bitCapInt fullMax = (ONE_BCI << primeBits) - 1U;
    const bitCapInt minPrime = primeDict[primeBits].size() ? primeDict[primeBits][0] : (fullMin + 1U);
    const bitCapInt minPrime2 = primeDict[primeBits].size() ? primeDict[primeBits][1] : (fullMin + 3U);
    const bitCapInt maxPrime = primeDict[primeBits].size() ? primeDict[primeBits][3] : fullMax;
    const bitCapInt maxPrime2 = primeDict[primeBits].size() ? primeDict[primeBits][2] : (fullMax - 2U);
    const bitCapInt minR = (toFactor / maxPrime - 1U) * (toFactor / maxPrime2 - 1U);
    const bitCapInt maxR = (toFactor / minPrime - 1U) * (toFactor / minPrime2 - 1U);
#else
    // \phi(n) is Euler's totient for n. A loose lower bound is \phi(n) >= sqrt(n/2).
    const bitCapInt minR = floorSqrt(toFactor >> 1U);
    // A better bound is \phi(n) >= pow(n / 2, log(2)/log(3))
    // const bitCapInt minR = pow(toFactor / 2, PHI_EXPONENT);

    // It can be shown that the period of this modular exponentiation can be no higher than 1
    // less than the modulus, as in https://www2.math.upenn.edu/~mlazar/math170/notes06-3.pdf.
    // Further, an upper bound on Euler's totient for composite numbers is n - sqrt(n). (See
    // https://math.stackexchange.com/questions/896920/upper-bound-for-eulers-totient-function-on-composite-numbers)
    const bitCapInt maxR = toFactor - floorSqrt(toFactor);
#endif

    std::vector<std::future<void>> futures(cpuCount);
    for (unsigned cpu = 0U; cpu < cpuCount; cpu++) {
        futures[cpu] = std::async(std::launch::async,
            [cpu, nodeId, nodeCount, toFactor, minR, maxR, &primeDict, &iterClock, &rand_gen, &isFinished] {
                // These constants are semi-redundant, but they're only defined once per thread,
                // and compilers differ on lambda expression capture of constants.

                // Batching reduces mutex-waiting overhead, on the std::atomic broadcast.
                // Batch size is BASE_TRIALS * PERIOD_TRIALS.

                // Number of times to reuse a random base:
                const size_t BASE_TRIALS = 1U;
                // Number of random period guesses per random base:
                const size_t PERIOD_TRIALS = 1U << 6U;

                const double clockFactor = 1.0 / 1000.0; // Report in ms
                const unsigned threads = std::thread::hardware_concurrency();

                const bitCapInt fullRange = maxR + 1U - minR;
                const bitCapInt nodeRange = fullRange / nodeCount;
                const bitCapInt nodeMin = minR + nodeRange * nodeId;
                const bitCapInt nodeMax = ((nodeId + 1U) == nodeCount) ? maxR : (minR + nodeRange * (nodeId + 1U) - 1U);
                const bitCapInt threadRange = (nodeMax + 1U - nodeMin) / threads;
                const bitCapInt baseMin = nodeMin + threadRange * cpu;
                const bitCapInt baseMax = ((cpu + 1U) == threads) ? nodeMax : (nodeMin + threadRange * (cpu + 1U) - 1U);

                std::vector<rand_dist> rDist;
#if QBCAPPOW < 7U
                rDist.push_back(rand_dist(baseMin, baseMax));
#else
                const bitLenInt wordSize = 64U;
                const bitCapInt wordMask = 0xFFFFFFFFFFFFFFFF;
                bitCapInt distPart = baseMax - baseMin;
                while (distPart) {
                    rDist.push_back(rand_dist(0U, (uint64_t)(distPart & wordMask)));
                    distPart >>= wordSize;
                }
                std::reverse(rDist.begin(), rDist.end());
#endif

                for (;;) {
                    for (size_t batchItem = 0U; batchItem < BASE_TRIALS; batchItem++) {
                        // Choose a base at random, >1 and <toFactor.
                        bitCapInt base = rDist[0U](rand_gen);
#if QBCAPPOW > 6U
                        for (size_t i = 1U; i < rDist.size(); i++) {
                            base <<= wordSize;
                            base |= rDist[i](rand_gen);
                        }
                        base += baseMin;
#endif

                        const bitCapInt testFactor = gcd(toFactor, base);
                        if (testFactor != 1U) {
                            // Inform the other threads on this node that we've succeeded and are done:
                            isFinished = true;

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
                        // const bitCapInt logBaseToFactor = (bitCapInt)intLog(base, toFactor) + 1U;
                        // Euler's Theorem tells us, if gcd(a, n) = 1, then a^\phi(n) = 1 MOD n,
                        // where \phi(n) is Euler's totient for n.
                        // const bitCapInt minR = (minPhi < logBaseToFactor) ? logBaseToFactor : minPhi;

                        // c is basically a harmonic degeneracy factor, and there might be no value in testing
                        // any case except c = 1, without loss of generality.

                        // This sets a nonuniform distribution on our y values to test.
                        // y values are close to qubitPower / rGuess, and we midpoint round.

                        // However, results are better with uniformity over r, rather than y.

                        // So, we guess r, between minR and maxR.
                        for (size_t rTrial = 0U; rTrial < PERIOD_TRIALS; rTrial++) {
                            // Choose a base at random, >1 and <toFactor.
                            bitCapInt r = rDist[0U](rand_gen);
#if QBCAPPOW > 6U
                            for (size_t i = 1U; i < rDist.size(); i++) {
                                r <<= wordSize;
                                r |= rDist[i](rand_gen);
                            }
                            r += baseMin;
#endif
                            // Since our output is r rather than y, we can skip the continued fractions step.
                            const bitCapInt p = (r & 1U) ? r : (r >> 1U);

#define PRINT_SUCCESS(f1, f2, toFactor)                                                                                \
    std::cout << "Success: Found " << (f1) << " * " << (f2) << " = " << (toFactor) << std::endl;                       \
    auto tClock =                                                                                                      \
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - iterClock);  \
    std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;                            \
    std::cout << "(Waiting to join other threads...)" << std::endl;

#if IS_RSA_SEMIPRIME
#define RGUESS p
#else
#define RGUESS r
#endif

                            // As a "classical" optimization, since \phi(toFactor) and factor bounds overlap,
                            // we first check if our guess for r is already a factor.
                            if ((RGUESS > 1U) && (((toFactor / RGUESS) * RGUESS) == toFactor)) {
                                // Inform the other threads on this node that we've succeeded and are done:
                                isFinished = true;

                                PRINT_SUCCESS(RGUESS, toFactor / RGUESS, toFactor);
                                return;
                            }

                            const bitCapInt apowrhalf = uipow(base, p) % toFactor;
                            bitCapInt f1 = (bitCapInt)gcd(apowrhalf + 1U, toFactor);
                            bitCapInt f2 = (bitCapInt)gcd(apowrhalf - 1U, toFactor);
                            bitCapInt fmul = f1 * f2;
                            while ((fmul != toFactor) && (fmul > 1U) && (((toFactor / fmul) * fmul) == toFactor)) {
                                fmul = f1;
                                f1 = fmul * f2;
                                f2 = toFactor / (fmul * f2);
                                fmul = f1 * f2;
                            }
                            if ((fmul == toFactor) && (f1 > 1U) && (f2 > 1U)) {
                                // Inform the other threads on this node that we've succeeded and are done:
                                isFinished = true;

                                PRINT_SUCCESS(f1, f2, toFactor);
                                return;
                            }
                        }
                    }

                    // Check if finished, between batches.
                    if (isFinished) {
                        return;
                    }
                }
            });
    };

    for (unsigned cpu = 0U; cpu < cpuCount; cpu++) {
        futures[cpu].get();
    }

    return 0;
}
