//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "qfactory.hpp"

#include "tests.hpp"

using namespace Qrack;

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1 __tmp_a = A;                                                                                             \
        real1 __tmp_b = B;                                                                                             \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                                                                        \
    } while (0);

const bitLenInt MaxQubits = 28;

void benchmarkLoopVariable(std::function<void(QInterfacePtr, int)> fn, bitLenInt mxQbts)
{

    const int ITERATIONS = 100;

    std::cout << std::endl;
    std::cout << ITERATIONS << " iterations";
    std::cout << std::endl;
    std::cout << "# of Qubits, ";
    std::cout << "Average Error, ";
    std::cout << "Sample Std. Deviation, ";
    std::cout << "Most accurate, ";
    std::cout << "1st Quartile, ";
    std::cout << "Median, ";
    std::cout << "3rd Quartile, ";
    std::cout << "Worst" << std::endl;

    std::complex<double> testAmp1, testAmp2;
    double err;
    double trialErrors[ITERATIONS];

    int i, numBits;

    double avge, stdee;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    for (numBits = 1; numBits <= mxQbts; numBits++) {
        QInterfacePtr qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, numBits,
            0, rng, complex(ONE_R1, ZERO_R1), !disable_normalization);
        avge = 0;
        for (i = 0; i < ITERATIONS; i++) {
            qftReg->SetPermutation(0);
            testAmp1 = qftReg->GetAmplitude(0);

            // Run loop body
            fn(qftReg, numBits);

            testAmp2 = qftReg->GetAmplitude(0);

            err = norm(testAmp1 - testAmp2);

            // Collect interval data
            trialErrors[i] = err;
            avge += err;
        }
        avge /= ITERATIONS;

        stdee = 0.0;
        for (i = 0; i < ITERATIONS; i++) {
            stdee += (trialErrors[i] - avge) * (trialErrors[i] - avge);
        }
        stdee = sqrt(stdee / ITERATIONS);

        std::sort(trialErrors, trialErrors + ITERATIONS);

        std::cout << (int)numBits << ", "; /* # of Qubits */
        std::cout << avge << ","; /* Average */
        std::cout << stdee << ","; /* Sample Std. Deviation */
        std::cout << trialErrors[0] << ","; /* Best */
        if (ITERATIONS % 4 == 0) {
            std::cout << (trialErrors[ITERATIONS / 4 - 1] + trialErrors[ITERATIONS / 4])
                      << ","; /* 1st Quartile */
        } else {
            std::cout << trialErrors[ITERATIONS / 4 - 1] << ","; /* 1st Quartile */
        }
        if (ITERATIONS % 2 == 0) {
            std::cout << (trialErrors[ITERATIONS / 2 - 1] + trialErrors[ITERATIONS / 2])
                      << ","; /* Median */
        } else {
            std::cout << trialErrors[ITERATIONS / 2 - 1] << ","; /* Median */
        }
        if (ITERATIONS % 4 == 0) {
            std::cout << (trialErrors[(3 * ITERATIONS) / 4 - 1] + trialErrors[(3 * ITERATIONS) / 4])
                      << ","; /* 3rd Quartile */
        } else {
            std::cout << trialErrors[(3 * ITERATIONS) / 4 - 1]
                      << ","; /* 3rd Quartile */
        }
        std::cout << trialErrors[ITERATIONS - 1] << std::endl; /* Worst */
    }
}

void benchmarkLoop(std::function<void(QInterfacePtr, int)> fn) { benchmarkLoopVariable(fn, MaxQubits); }

TEST_CASE("test_h")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) {
        qftReg->H(0);
        qftReg->H(0);
    });
}
