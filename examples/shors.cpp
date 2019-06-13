//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This example demonstrates Shor's algorithm for integer factoring.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <cmath>
#include <iomanip> // For setw
#include <iostream> // For cout
#include <stdlib.h>
#include <time.h>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

bitCapInt gcd(bitCapInt n1, bitCapInt n2)
{
    while (n1 != n2) {
        if (n1 > n2)
            n1 -= n2;
        else
            n2 -= n1;
    }

    return n1;
}

int main()
{
    bitCapInt toFactor, base;

    std::cout << "Number to factor: ";
    std::cin >> toFactor;

    bitLenInt qubitCount = ceil(log2(toFactor));

    // Choose a base at random:
    srand(time(NULL));
    base = (bitCapInt)(rand() % toFactor + 1U);

    bitCapInt testFactor = gcd(toFactor, base);
    if (testFactor == 1) {
        std::cout << "Lucky guess chose non- relative prime: " << testFactor;
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
    bitCapInt r = qReg->MReg(0, qubitCount);

    // Try to determine the factors
    if (r & 1U) {
        r *= 2;
    }
    bitCapInt apowrhalf = (int)pow(base, r >> 1) % toFactor;
    bitCapInt f1 = gcd(apowrhalf + 1, toFactor);
    bitCapInt f2 = gcd(apowrhalf - 1, toFactor);
    bitCapInt res1 = f1;
    bitCapInt res2 = f2;
    if (((f1 * f2) != toFactor) && ((f1 * f2) > 1) && ((int)((ONE_R1 * toFactor) / (f1 * f2)) * f1 * f2 == toFactor)) {
        res1 = f1 * f2;
        res2 = toFactor / (f1 * f2);
    }
    if (((res1 * res2) == toFactor) && (res1 > 1) && (res2 > 1)) {
        std::cout << "Success: Found " << res1 << " * " << f2 << " = " << toFactor << std::endl;
    } else {
        std::cout << "Failure: Found " << res1 << " and " << f2 << std::endl;
    }
}
