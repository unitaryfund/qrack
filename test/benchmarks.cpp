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

const bitLenInt MaxQubits = 24;

const double clockFactor = 1000.0 / CLOCKS_PER_SEC; // Report in ms

void benchmarkLoopVariable(std::function<void(QInterfacePtr, int)> fn, bitLenInt mxQbts, bool resetRandomPerm = true,
    bool hadamardRandomBits = false)
{

    const int ITERATIONS = 100;

    std::cout << std::endl;
    std::cout << ITERATIONS << " iterations";
    std::cout << std::endl;
    std::cout << "# of Qubits, ";
    std::cout << "Average Time (ms), ";
    std::cout << "Sample Std. Deviation (ms), ";
    std::cout << "Fastest (ms), ";
    std::cout << "1st Quartile (ms), ";
    std::cout << "Median (ms), ";
    std::cout << "3rd Quartile (ms), ";
    std::cout << "Slowest (ms)" << std::endl;

    clock_t tClock, iterClock;
    real1 trialClocks[ITERATIONS];

    int i, j, numBits;

    real1 avgt, stdet;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    for (numBits = 4; numBits <= mxQbts; numBits++) {
        QInterfacePtr qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, numBits,
            0, rng, complex(ONE_R1, ZERO_R1), enable_normalization);
        avgt = 0.0;

        for (i = 0; i < ITERATIONS; i++) {
            if (resetRandomPerm) {
                qftReg->SetPermutation(qftReg->Rand() * qftReg->GetMaxQPower());
            } else {
                qftReg->SetPermutation(0);
            }
            if (hadamardRandomBits) {
                for (j = 0; j < numBits; j++) {
                    if (qftReg->Rand() >= ONE_R1 / 2) {
                        qftReg->H(j);
                    }
                }
            }
            qftReg->Finish();

            iterClock = clock();

            // Run loop body
            fn(qftReg, numBits);

            if (!async_time) {
                qftReg->Finish();
            }

            // Collect interval data
            tClock = clock() - iterClock;
            trialClocks[i] = log2(tClock * clockFactor);
            avgt += trialClocks[i];

            if (async_time) {
                qftReg->Finish();
            }
        }
        avgt /= ITERATIONS;

        stdet = 0.0;
        for (i = 0; i < ITERATIONS; i++) {
            stdet += (trialClocks[i] - avgt) * (trialClocks[i] - avgt);
        }
        stdet = sqrt(stdet / ITERATIONS);

        std::sort(trialClocks, trialClocks + ITERATIONS);

        std::cout << (int)numBits << ", "; /* # of Qubits */
        std::cout << pow(2.0, avgt) << ","; /* Average Time (ms) */
        std::cout << pow(2.0, stdet) << ","; /* Sample Std. Deviation (ms) */
        std::cout << pow(2.0, trialClocks[0]) << ","; /* Fastest (ms) */
        if (ITERATIONS % 4 == 0) {
            std::cout << pow(2.0, (trialClocks[ITERATIONS / 4 - 1] + trialClocks[ITERATIONS / 4]) / 2)
                      << ","; /* 1st Quartile (ms) */
        } else {
            std::cout << pow(2.0, trialClocks[ITERATIONS / 4 - 1] / 2) << ","; /* 1st Quartile (ms) */
        }
        if (ITERATIONS % 2 == 0) {
            std::cout << pow(2.0, (trialClocks[ITERATIONS / 2 - 1] + trialClocks[ITERATIONS / 2]) / 2)
                      << ","; /* Median (ms) */
        } else {
            std::cout << pow(2.0, trialClocks[ITERATIONS / 2 - 1] / 2) << ","; /* Median (ms) */
        }
        if (ITERATIONS % 4 == 0) {
            std::cout << pow(2.0, (trialClocks[(3 * ITERATIONS) / 4 - 1] + trialClocks[(3 * ITERATIONS) / 4]) / 2)
                      << ","; /* 3rd Quartile (ms) */
        } else {
            std::cout << pow(2.0, trialClocks[(3 * ITERATIONS) / 4 - 1] / 2) << ","; /* 3rd Quartile (ms) */
        }
        std::cout << pow(2.0, trialClocks[ITERATIONS - 1]) << std::endl; /* Slowest (ms) */
    }
}

void benchmarkLoop(
    std::function<void(QInterfacePtr, int)> fn, bool resetRandomPerm = true, bool hadamardRandomBits = false)
{
    benchmarkLoopVariable(fn, MaxQubits, resetRandomPerm, hadamardRandomBits);
}

TEST_CASE("test_cnot_single")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CNOT(0, 1, 1); });
}

TEST_CASE("test_x_single")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->X(0, 1); });
}

TEST_CASE("test_swap_single")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Swap(0, 1); });
}

TEST_CASE("test_cnot_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CNOT(0, n / 2, n / 2); });
}

TEST_CASE("test_ccnot_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CCNOT(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_swap_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Swap(0, n / 2, n / 2); });
}

TEST_CASE("test_x_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->X(0, n); });
}

TEST_CASE("test_and_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->AND(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_or_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->OR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_xor_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->XOR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_cland_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLAND(0, 0x0c, 0, n); });
}

TEST_CASE("test_clor_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_clxor_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLXOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_rt_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->RT(M_PI, 0, n); });
}

TEST_CASE("test_crt_all")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CRT(M_PI, 0, n / 2, n / 2); });
}

TEST_CASE("test_rol")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ROL(1, 0, n); });
}

TEST_CASE("test_inc")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INC(1, 0, n); });
}

TEST_CASE("test_incs")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCS(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incc")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCC(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incsc")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCSC(1, 0, n - 2, n - 2, n - 1); });
}

TEST_CASE("test_zero_phase_flip")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ZeroPhaseFlip(0, n); });
}

TEST_CASE("test_c_phase_flip_if_less")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CPhaseFlipIfLess(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_phase_flip")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->PhaseFlip(); });
}

TEST_CASE("test_m")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->M(n - 1); });
}

TEST_CASE("test_mreg")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->MReg(0, n); });
}

void benchmarkSuperpose(std::function<void(QInterfacePtr, int, unsigned char*)> fn)
{
    bitCapInt i, j;

    bitCapInt wordLength = (MaxQubits / 16 + 1);
    bitCapInt indexLength = (1 << (MaxQubits / 2));
    unsigned char* testPage = new unsigned char[wordLength * indexLength];
    for (j = 0; j < indexLength; j++) {
        for (i = 0; i < wordLength; i++) {
            testPage[j * wordLength + i] = (j & (0xff << (8 * i))) >> (8 * i);
        }
    }
    benchmarkLoop([fn, testPage](QInterfacePtr qftReg, int n) { fn(qftReg, n, testPage); });
    delete[] testPage;
}

TEST_CASE("test_superposition_reg")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedLDA(0, n / 2, n / 2, n / 2, testPage);
    });
}

TEST_CASE("test_adc_superposition_reg")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedADC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_sbc_superposition_reg")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedSBC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_setbit")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->SetBit(0, true); });
}

TEST_CASE("test_proball")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ProbAll(0x02); });
}

TEST_CASE("test_set_reg")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->SetReg(0, n, 1); });
}

TEST_CASE("test_grover")
{

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 3.

    benchmarkLoopVariable(
        [](QInterfacePtr qftReg, int n) {
            int i;
            // Twelve iterations maximizes the probablity for 256 searched elements, for example.
            // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
            int optIter = M_PI / (4.0 * asin(1.0 / sqrt(1 << n)));

            // Our input to the subroutine "oracle" is 8 bits.
            qftReg->SetPermutation(0);
            qftReg->H(0, n);

            for (i = 0; i < optIter; i++) {
                // Our "oracle" is true for an input of "3" and false for all other inputs.
                qftReg->DEC(3, 0, n);
                qftReg->ZeroPhaseFlip(0, n);
                qftReg->INC(3, 0, n);
                // This ends the "oracle."
                qftReg->H(0, n);
                qftReg->ZeroPhaseFlip(0, n);
                qftReg->H(0, n);
                qftReg->PhaseFlip();
            }

            REQUIRE_THAT(qftReg, HasProbability(0x3));

            qftReg->MReg(0, n);
        },
        16);
}

TEST_CASE("test_qft_ideal_init")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->QFT(0, n, false); }, false, false);
}

TEST_CASE("test_qft_permutation_init")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->QFT(0, n, false); }, true, false);
}

TEST_CASE("test_qft_permutation_round_trip_entangled")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, false);
}

TEST_CASE("test_qft_permutation_round_trip_separated")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, true);
        },
        true, false);
}

TEST_CASE("test_qft_superposition_round_trip")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, true);
}
