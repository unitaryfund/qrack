//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This example demonstrates Shor's algorithm for integer factoring. (This file was heavily adapted from
// https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/shor.py, with thanks to ProjectQ!)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <cmath>
#include <iomanip> // For setw
#include <iostream> // For cout
#include <random>
#include <stdlib.h>
#include <time.h>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

bitCapInt gcd(bitCapInt n1, bitCapInt n2)
{
    if (n2 == 0)
        return n1;
    return gcd(n2, n1 % n2);
}

bitCapInt continued_fraction_step(bitCapInt* numerator, bitCapInt* denominator)
{
    bitCapInt intPart = (*numerator) / (*denominator);
    bitCapInt partDenominator = (*numerator) - intPart * (*denominator);
    bitCapInt partNumerator = (*denominator);

    (*numerator) = partNumerator;
    (*denominator) = partDenominator;
    return intPart;
}

real1 calc_continued_fraction(std::vector<bitCapInt> denominators, bitCapInt* numerator, bitCapInt* denominator)
{
    bitCapInt approxNumer = 1;
    bitCapInt approxDenom = denominators.back();
    bitCapInt temp;

    for (int i = (denominators.size() - 1); i > 0; i--) {
        temp = denominators[i] * approxDenom + approxNumer;
        approxNumer = approxDenom;
        approxDenom = temp;
    }

    (*numerator) = approxNumer;
    (*denominator) = approxDenom;
    return ((real1)approxNumer) / approxDenom;
}

int main()
{
    uint64_t toFactor;
    bitCapInt base;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

    bitLenInt qubitCount = ceil(log2(toFactor));

    // Choose a base at random:
    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());
    std::uniform_int_distribution<> rand_dist(2, toFactor - 1);
    base = (bitCapInt)(rand_dist(rand_gen));

    bitCapInt testFactor = gcd(toFactor, base);
    if (testFactor != 1) {
        std::cout << "Chose non- relative prime: " << (uint64_t)testFactor << " * " << (uint64_t)(toFactor / testFactor) << std::endl;
        return 0;
    }

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, qubitCount * 4, 0);

    // This is the period-finding subroutine of Shor's algorithm.
    qReg->H(0, qubitCount);
    qReg->POWModNOut(base, toFactor, 0, qubitCount, qubitCount);
    qReg->IQFT(0, qubitCount);

    bitCapInt y = qReg->MReg(0, qubitCount);
    if (y == 0) {
        std::cout << "Failed: y = 0 in period estimation subroutine." << std::endl;
        return 0;
    }
    bitCapInt qubitPower = 1U << qubitCount;

    // Value is always fractional, so skip first step, by flipping numerator and denominator:
    bitCapInt numerator = qubitPower;
    bitCapInt denominator = y;

    std::vector<bitCapInt> denominators;
    bitCapInt approxNumer;
    bitCapInt approxDenom;
    do {
        denominators.push_back(continued_fraction_step(&numerator, &denominator));
        calc_continued_fraction(denominators, &approxNumer, &approxDenom);
    } while ((denominator > 0) && (approxDenom < toFactor));
    denominators.pop_back();

    bitCapInt r;
    if (denominators.size() == 0) {
        r = y;
    } else {
        calc_continued_fraction(denominators, &approxNumer, &r);
    }

    // Try to determine the factors
    if (r & 1U) {
        r *= 2;
    }
    bitCapInt apowrhalf = ((bitCapInt)pow((double)base, (double)(r >> 1U))) % toFactor;
    bitCapInt f1 = gcd(apowrhalf + 1, toFactor);
    bitCapInt f2 = gcd(apowrhalf - 1, toFactor);
    bitCapInt res1 = f1;
    bitCapInt res2 = f2;
    if (((f1 * f2) != toFactor) && ((f1 * f2) > 1) &&
        (((int)((ONE_R1 * toFactor) / (f1 * f2)) * f1 * f2) == toFactor)) {
        res1 = f1 * f2;
        res2 = toFactor / (f1 * f2);
    }
    if (((res1 * res2) == toFactor) && (res1 > 1) && (res2 > 1)) {
        std::cout << "Success: Found " << (uint64_t)res1 << " * " << (uint64_t)res2 << " = " << (uint64_t)toFactor << std::endl;
    } else {
        std::cout << "Failure: Found " << (uint64_t)res1 << " and " << (uint64_t)res2 << std::endl;
    }

    return 0;
}
