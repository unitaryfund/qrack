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

#include <algorithm>
#include <atomic>
#include <future>
#include <map>
#include <mutex>

// Turn this off, if you're not factoring a semi-prime number with equal-bit-width factors.
#define IS_RSA_SEMIPRIME 1
// Turn this off, if you don't want to coordinate across multiple (quasi-independent) nodes.
#define IS_DISTRIBUTED 1
// The maximum number of bits in Boost big integers is 2^QBCAPPOW.
// (2^7, only, needs custom std::cout << operator implementation.)
#define QBCAPPOW 7U

#if QBCAPPOW < 32U
#define bitLenInt uint32_t
#else
#define bitLenInt uint64_t
#endif

#if QBCAPPOW < 6U
#define bitCapInt uint32_t
#define ONE_BCI 1UL
#elif QBCAPPOW < 7U
#define bitCapInt uint64_t
#define ONE_BCI 1ULL
#elif QBCAPPOW < 8U
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt boost::multiprecision::uint128_t
#define ONE_BCI 1ULL
#else
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1ULL << QBCAPPOW, 1ULL << QBCAPPOW,           \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#define ONE_BCI 1ULL
#endif

#define WORD uint64_t
#define WORD_SIZE 64U

namespace Qimcifa {

#if QBCAPPOW == 7U
std::ostream& operator<<(std::ostream& os, bitCapInt b)
{
    // Calculate the base-10 digits, from lowest to highest.
    std::vector<std::string> digits;
    while (b) {
        digits.push_back(std::to_string((unsigned char)(b % 10U)));
        b /= 10U;
    }

    // Reversing order, print the digits from highest to lowest.
    for (size_t i = digits.size() - 1U; i > 0; --i) {
        os << digits[i];
    }
    // Avoid the need for a signed comparison.
    os << digits[0];

    return os;
}

std::istream& operator>>(std::istream& is, bitCapInt& b)
{
    // Get the whole input string at once.
    std::string input;
    is >> input;

    // Start the output address value at 0.
    b = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        // Left shift by 1 base-10 digit.
        b *= 10;
        // Add the next lowest base-10 digit.
        b += (input[i] - 48U);
    }

    return is;
}
#endif

// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x) { return (x && !(x & (x - ONE_BCI))); }

inline bitLenInt log2(const bitCapInt& n)
{
#if __GNUC__ && QBCAPPOW < 7
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)((sizeof(unsigned int) << 3U) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)((sizeof(unsigned long long) << 3U) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
    bitLenInt pow = 0U;
    bitCapInt p = n >> 1U;
    while (p) {
        p >>= 1U;
        ++pow;
    }
    return pow;
#endif
}

// Source:
// https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int#answer-101613
bitCapInt uipow(bitCapInt base, bitCapInt exp)
{
    bitCapInt result = 1U;
    for (;;) {
        if (base & 1U) {
            result *= base;
        }
        exp >>= 1U;
        if (!exp) {
            break;
        }
        base *= base;
    }

    return result;
}

// Adapted from Gaurav Ahirwar's suggestion on https://www.geeksforgeeks.org/square-root-of-an-integer/
bitCapInt floorSqrt(const bitCapInt& x)
{
    // Base cases
    if ((x == 0U) || (x == 1U)) {
        return x;
    }

    // Binary search for floor(sqrt(x))
    bitCapInt start = 1U, end = x >> 1U, ans = 0U;
    do {
        const bitCapInt mid = (start + end) >> 1U;

        // If x is a perfect square
        const bitCapInt sqr = mid * mid;
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
    } while (start <= end);

    return ans;
}

bitCapInt gcd(bitCapInt n1, bitCapInt n2)
{
    while (n2) {
        const bitCapInt t = n1;
        n1 = n2;
        n2 = t % n2;
    }

    return n1;
}

} // namespace Qimcifa

using namespace Qimcifa;

int main()
{
    typedef std::uniform_int_distribution<WORD> rand_dist;

    bitCapInt toFactor;
    bitCapInt nodeCount = 1U;
    bitCapInt nodeId = 0U;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

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

    auto iterClock = std::chrono::high_resolution_clock::now();

    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());

    const unsigned cpuCount = std::thread::hardware_concurrency();
    std::atomic<bool> isFinished;
    isFinished = false;

#if IS_RSA_SEMIPRIME
    std::map<bitLenInt, const std::vector<bitCapInt>> primeDict = { { 16U, { 32771U, 65521U } },
        { 28U, { 134217757U, 268435399U } }, { 32U, { 2147483659U, 4294967291U } },
        { 64U, { 9223372036854775837U, 1844674407370955143U } } };

    // If n is semiprime, \phi(n) = (p - 1) * (q - 1), where "p" and "q" are prime.
    // The minimum value of this formula, for our input, without consideration of actual
    // primes in the interval, is as follows:
    // (See https://www.mobilefish.com/services/rsa_key_generation/rsa_key_generation.php)
    const bitLenInt primeBits = (qubitCount + 1U) >> 1U;
    const bitCapInt fullMin = ONE_BCI << (primeBits - 1U);
    const bitCapInt fullMax = (ONE_BCI << primeBits) - 1U;
    const bitCapInt minPrime = primeDict[primeBits].size() ? primeDict[primeBits][0] : (fullMin + 1U);
    const bitCapInt maxPrime = primeDict[primeBits].size() ? primeDict[primeBits][1] : fullMax;
    const bitCapInt fullMinR = (minPrime - 1U) * (toFactor / minPrime - 1U);
    const bitCapInt fullMaxR = (maxPrime - 1U) * (toFactor / maxPrime - 1U);
#else
    // \phi(n) is Euler's totient for n. A loose lower bound is \phi(n) >= sqrt(n/2).
    const bitCapInt fullMinR = floorSqrt(toFactor >> 1U);
    // A better bound is \phi(n) >= pow(n / 2, log(2)/log(3))
    // const bitCapInt fullMinR = pow(toFactor / 2, PHI_EXPONENT);

    // It can be shown that the period of this modular exponentiation can be no higher than 1
    // less than the modulus, as in https://www2.math.upenn.edu/~mlazar/math170/notes06-3.pdf.
    // Further, an upper bound on Euler's totient for composite numbers is n - sqrt(n). (See
    // https://math.stackexchange.com/questions/896920/upper-bound-for-eulers-totient-function-on-composite-numbers)
    const bitCapInt fullMaxR = toFactor - floorSqrt(toFactor);
#endif

    std::vector<rand_dist> baseDist;
#if QBCAPPOW > 6U
    bitCapInt distPart = toFactor - 3U;
    while (distPart) {
        baseDist.push_back(rand_dist(0U, (WORD)distPart));
        distPart >>= WORD_SIZE;
    }
    std::reverse(baseDist.begin(), baseDist.end());
#else
    baseDist.push_back(rand_dist(2U, toFactor - 1U));
#endif

    auto workerFn = [&nodeId, &nodeCount, &toFactor, &fullMinR, &fullMaxR, &baseDist, &iterClock, &rand_gen,
                        &isFinished](int cpu) {
        // These constants are semi-redundant, but they're only defined once per thread,
        // and compilers differ on lambda expression capture of constants.

        // Batching reduces mutex-waiting overhead, on the std::atomic broadcast.
        // Batch size is BASE_TRIALS * PERIOD_TRIALS.

        // Number of times to reuse a random base:
        const size_t BASE_TRIALS = 1U << 4U;

        const double clockFactor = 1.0 / 1000.0; // Report in ms
        const unsigned threads = std::thread::hardware_concurrency();

        const bitCapInt fullRange = fullMaxR + 1U - fullMinR;
        const bitCapInt nodeRange = fullRange / nodeCount;
        const bitCapInt nodeMin = fullMinR + nodeRange * nodeId;
        const bitCapInt nodeMax = ((nodeId + 1U) == nodeCount) ? fullMaxR : (fullMinR + nodeRange * (nodeId + 1U) - 1U);
        const bitCapInt threadRange = (nodeMax + 1U - nodeMin) / threads;
        const bitCapInt rMin = nodeMin + threadRange * cpu;
        const bitCapInt rMax = ((cpu + 1U) == threads) ? nodeMax : (nodeMin + threadRange * (cpu + 1U) - 1U);

        std::vector<rand_dist> rDist;
#if QBCAPPOW > 6U
#if IS_RSA_SEMIPRIME
        // Euler's totient is the product of 2 even numbers, so it is a multiple of 4.
        bitCapInt distPart = (rMax - rMin) >> 2U;
#else
        bitCapInt distPart = rMax - rMin;
#endif
        while (distPart) {
            rDist.push_back(rand_dist(0U, (WORD)distPart));
            distPart >>= WORD_SIZE;
        }
        std::reverse(rDist.begin(), rDist.end());
#elif IS_RSA_SEMIPRIME
        // Euler's totient is the product of 2 even numbers, so it is a multiple of 4.
        rDist.push_back(rand_dist(rMin >> 2U, rMax >> 2U));
#else
        rDist.push_back(rand_dist(rMin, rMax));
#endif

        for (;;) {
            for (size_t batchItem = 0U; batchItem < BASE_TRIALS; ++batchItem) {
                // Choose a base at random, >1 and <toFactor.
                bitCapInt base = baseDist[0U](rand_gen);
#if QBCAPPOW > 6U
                for (size_t i = 1U; i < baseDist.size(); ++i) {
                    base <<= WORD_SIZE;
                    base |= baseDist[i](rand_gen);
                }
                base += 2U;
#endif

#define PRINT_SUCCESS(f1, f2, toFactor, message)                                                                       \
    std::cout << message << (f1) << " * " << (f2) << " = " << (toFactor) << std::endl;                                 \
    auto tClock =                                                                                                      \
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - iterClock);  \
    std::cout << "(Time elapsed: " << (tClock.count() * clockFactor) << "ms)" << std::endl;                            \
    std::cout << "(Waiting to join other threads...)" << std::endl;

                const bitCapInt testFactor = gcd(toFactor, base);
                if (testFactor != 1U) {
                    // Inform the other threads on this node that we've succeeded and are done:
                    isFinished = true;

                    PRINT_SUCCESS(testFactor, (toFactor / testFactor), toFactor, "Chose non-relative prime: Found ");
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
                // Euler's Theorem tells us, if gcd(a, n) = 1, then a^\phi(n) = 1 MOD n,
                // where \phi(n) is Euler's totient for n.

                // c is basically a harmonic degeneracy factor, and there might be no value in testing
                // any case except c = 1, without loss of generality.

                // This sets a nonuniform distribution on our y values to test.
                // y values are close to qubitPower / rGuess, and we midpoint round.

                // However, results are better with uniformity over r, rather than y.

                // So, we guess r, between fullMinR and fullMaxR.
                // Since our output is r rather than y, we can skip the continued fractions step.
                bitCapInt r = rDist[0U](rand_gen);
#if QBCAPPOW > 6U
                for (size_t i = 1U; i < rDist.size(); ++i) {
                    r <<= WORD_SIZE;
                    r |= rDist[i](rand_gen);
                }
#if IS_RSA_SEMIPRIME
                // Euler's totient is the product of 2 even numbers, so it is a multiple of 4.
                r += rMin >> 2U;
#else
                r += rMin;
#endif
#endif

#if IS_RSA_SEMIPRIME
                // Euler's totient is the product of 2 even numbers, so it is a multiple of 4.
                r <<= 2U;
#else
                if (r & 1U) {
                    r <<= 1U;
                }

                // As a "classical" optimization, since \phi(toFactor) and factor bounds overlap,
                // we first check if our guess for r is already a factor.
                const bitCapInt testFactor = gcd(toFactor, r);
                if (testFactor != 1U) {
                    // Inform the other threads on this node that we've succeeded and are done:
                    isFinished = true;

                    PRINT_SUCCESS(testFactor, toFactor / testFactor, toFactor, "Success (on r trial division): Found ");
                    return;
                }
#endif

                const bitCapInt apowrhalf = uipow(base, r) % toFactor;
                bitCapInt f1 = gcd(apowrhalf + 1U, toFactor);
                bitCapInt f2 = gcd(apowrhalf - 1U, toFactor);
                bitCapInt fmul = f1 * f2;
                while ((fmul > 1U) && (fmul != toFactor) && ((toFactor % fmul) == 0)) {
                    fmul = f1;
                    f1 *= f2;
                    f2 = toFactor / (fmul * f2);
                    fmul = f1 * f2;
                }
                if ((fmul == toFactor) && (f1 > 1U) && (f2 > 1U)) {
                    // Inform the other threads on this node that we've succeeded and are done:
                    isFinished = true;

                    PRINT_SUCCESS(f1, f2, toFactor, "Success (on r difference of squares): Found ");
                    return;
                }
            }

            // Check if finished, between batches.
            if (isFinished) {
                return;
            }
        }
    };

    std::vector<std::future<void>> futures(cpuCount);
    for (unsigned cpu = 0U; cpu < cpuCount; ++cpu) {
        futures[cpu] = std::async(std::launch::async, workerFn, cpu);
    };

    for (unsigned cpu = 0U; cpu < cpuCount; ++cpu) {
        futures[cpu].get();
    }

    return 0;
}
