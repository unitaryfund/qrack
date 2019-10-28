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
#include <set>
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

const double clockFactor = 1000.0 / CLOCKS_PER_SEC; // Report in ms

double formatTime(double t, bool logNormal)
{
    if (logNormal) {
        return pow(2.0, t);
    } else {
        return t;
    }
}

void benchmarkLoopVariable(std::function<void(QInterfacePtr, int)> fn, bitLenInt mxQbts, bool resetRandomPerm = true,
    bool hadamardRandomBits = false, bool logNormal = false)
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

    bitLenInt i, j, numBits;

    double avgt, stdet;

    bitLenInt mnQbts;
    if (single_qubit_run) {
        mnQbts = mxQbts;
    } else {
        mnQbts = 4;
    }

    for (numBits = mnQbts; numBits <= mxQbts; numBits++) {

        QInterfacePtr qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, numBits,
            0, rng, complex(ONE_R1, ZERO_R1), enable_normalization, true, false, device_id, !disable_hardware_rng);
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
            if (logNormal) {
                trialClocks[i] = log2(tClock * clockFactor);
            } else {
                trialClocks[i] = tClock * clockFactor;
            }
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
        std::cout << formatTime(avgt, logNormal) << ","; /* Average Time (ms) */
        std::cout << formatTime(stdet, logNormal) << ","; /* Sample Std. Deviation (ms) */
        std::cout << formatTime(trialClocks[0], logNormal) << ","; /* Fastest (ms) */
        if (ITERATIONS % 4 == 0) {
            std::cout << formatTime((trialClocks[ITERATIONS / 4 - 1] + trialClocks[ITERATIONS / 4]) / 2, logNormal)
                      << ","; /* 1st Quartile (ms) */
        } else {
            std::cout << formatTime(trialClocks[ITERATIONS / 4 - 1] / 2, logNormal) << ","; /* 1st Quartile (ms) */
        }
        if (ITERATIONS % 2 == 0) {
            std::cout << formatTime((trialClocks[ITERATIONS / 2 - 1] + trialClocks[ITERATIONS / 2]) / 2, logNormal)
                      << ","; /* Median (ms) */
        } else {
            std::cout << formatTime(trialClocks[ITERATIONS / 2 - 1] / 2, logNormal) << ","; /* Median (ms) */
        }
        if (ITERATIONS % 4 == 0) {
            std::cout << formatTime(
                             (trialClocks[(3 * ITERATIONS) / 4 - 1] + trialClocks[(3 * ITERATIONS) / 4]) / 2, logNormal)
                      << ","; /* 3rd Quartile (ms) */
        } else {
            std::cout << formatTime(trialClocks[(3 * ITERATIONS) / 4 - 1] / 2, logNormal)
                      << ","; /* 3rd Quartile (ms) */
        }
        std::cout << formatTime(trialClocks[ITERATIONS - 1], logNormal) << std::endl; /* Slowest (ms) */
    }
}

void benchmarkLoop(std::function<void(QInterfacePtr, int)> fn, bool resetRandomPerm = true,
    bool hadamardRandomBits = false, bool logNormal = false)
{
    benchmarkLoopVariable(fn, max_qubits, resetRandomPerm, hadamardRandomBits, logNormal);
}

TEST_CASE("test_cnot_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CNOT(0, 1); });
}

TEST_CASE("test_x_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->X(0); });
}

TEST_CASE("test_y_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Y(0); });
}

TEST_CASE("test_z_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Z(0); });
}

TEST_CASE("test_swap_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Swap(0, 1); });
}

TEST_CASE("test_cnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CNOT(0, n / 2, n / 2); });
}

TEST_CASE("test_x_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->X(0, n); });
}

TEST_CASE("test_y_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Y(0, n); });
}

TEST_CASE("test_z_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Z(0, n); });
}

TEST_CASE("test_swap_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->Swap(0, n / 2, n / 2); });
}

TEST_CASE("test_ccnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CCNOT(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_and_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->AND(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_or_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->OR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_xor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->XOR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_cland_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLAND(0, 0x0c, 0, n); });
}

TEST_CASE("test_clor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_clxor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CLXOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_rt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->RT(M_PI, 0, n); });
}

TEST_CASE("test_crt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CRT(M_PI, 0, n / 2, n / 2); });
}

TEST_CASE("test_rol", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ROL(1, 0, n); });
}

TEST_CASE("test_inc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INC(1, 0, n); });
}

TEST_CASE("test_incs", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCS(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCC(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incsc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->INCSC(1, 0, n - 2, n - 2, n - 1); });
}

TEST_CASE("test_zero_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ZeroPhaseFlip(0, n); });
}

TEST_CASE("test_c_phase_flip_if_less", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->CPhaseFlipIfLess(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->PhaseFlip(); });
}

TEST_CASE("test_m", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->M(n - 1); });
}

TEST_CASE("test_mreg", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->MReg(0, n); });
}

void benchmarkSuperpose(std::function<void(QInterfacePtr, int, unsigned char*)> fn)
{
    bitCapInt i, j;

    bitCapInt wordLength = (max_qubits / 16 + 1);
    bitCapInt indexLength = (1 << (max_qubits / 2));
    unsigned char* testPage = new unsigned char[wordLength * indexLength];
    for (j = 0; j < indexLength; j++) {
        for (i = 0; i < wordLength; i++) {
            testPage[j * wordLength + i] = (j & (0xff << (8 * i))) >> (8 * i);
        }
    }
    benchmarkLoop([fn, testPage](QInterfacePtr qftReg, int n) { fn(qftReg, n, testPage); });
    delete[] testPage;
}

TEST_CASE("test_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedLDA(0, n / 2, n / 2, n / 2, testPage);
    });
}

TEST_CASE("test_adc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedADC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_sbc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, int n, unsigned char* testPage) {
        qftReg->IndexedSBC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_setbit", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->SetBit(0, true); });
}

TEST_CASE("test_proball", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->ProbAll(0x02); });
}

TEST_CASE("test_set_reg", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->SetReg(0, n, 1); });
}

TEST_CASE("test_grover", "[grover]")
{

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 3.

    benchmarkLoop([](QInterfacePtr qftReg, int n) {
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
    });
}

TEST_CASE("test_qft_ideal_init", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, int n) { qftReg->QFT(0, n, false); }, false, false);
}

TEST_CASE("test_qft_permutation_init", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) { qftReg->QFT(0, n, false); }, true, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_permutation_round_trip_entangled", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_one_way", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) { qftReg->QFT(0, n, false); }, true, true, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_round_trip", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, int n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, true, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_solved_circuit", "[supreme]")
{
    // This is a "solved circuit," in that it is "classically efficient."

    const int depth = 20;
    benchmarkLoop([](QInterfacePtr qReg, int n) {

        int rowLen = std::sqrt(n);
        while (((n / rowLen) * rowLen) != n) {
            rowLen--;
        }

        real1 gateRand;
        int b1, b2;
        bitLenInt i, d;

        // We repeat the entire prepartion for "depth" iterations.
        // At very low depths, with only nearest neighbor entanglement, we can avoid entangling the representation of
        // the entire state as a single Schr{\"o}dinger method unit.
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                gateRand = qReg->Rand();

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but "Sqrt" variants are
                // handled as general single bit gates.
                if (gateRand < (ONE_R1 / 3)) {
                    qReg->X(i);
                } else if (gateRand < (2 * ONE_R1 / 3)) {
                    qReg->Y(i);
                } else {
                    // H(i) would likely be preferable for generality of our algebra, but it does not yet commute
                    // efficiently with CZ.
                    qReg->Z(i);
                }
            }

            std::set<bitLenInt> usedBits;

            for (i = 0; i < (n - rowLen); i++) {
                if (usedBits.find(i) != usedBits.end()) {
                    continue;
                }

                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In the interior bulk, one 2 bit gate is applied for every pair of bits, (as many gates as 1/2 the
                // number of bits). (Unless n is a perfect square, the "row length" has to be factored into a
                // rectangular shape, and "n" is sometimes prime or factors awkwardly.)

                b1 = i;
                // Next row
                b2 = i + rowLen;

                if (qReg->Rand() < (ONE_R1 / 2)) {
                    // Next column
                    // (Stop at boundaries of rectangle)
                    if (((i / rowLen) % 2) == 0) {
                        if ((b2 % rowLen) > 0) {
                            b2--;
                        }
                    } else {
                        if ((b2 % rowLen) < (rowLen - 1)) {
                            b2++;
                        }
                    }
                }

                // "iSWAP" is read to be a SWAP operation that imparts a phase factor of i if the bits are
                // different. Our "SWAP" is much better optimized.
                qReg->Swap(b1, b2);
                // "1/6 of CZ" is read to indicate the 6th root. Our full CZ is much better optimized.
                qReg->CZ(b1, b2);
                // Note that these gates are both symmetric under exchange of "b1" and "b2".

                usedBits.insert(b1);
                usedBits.insert(b2);
            }
        }

        qReg->MReg(0, n);
    });
}

TEST_CASE("test_polynomial", "[supreme]")
{
    // This is the closest an author of qrack can come to an efficient simulation of the benchmark argued to show
    // quantum supremacy.

    const int depth = 20;
    benchmarkLoop([](QInterfacePtr qReg, int n) {

        int rowLen = std::sqrt(n);
        while (((n / rowLen) * rowLen) != n) {
            rowLen--;
        }

        real1 gateRand;
        int b1, b2;
        bitLenInt i, d;
        bitLenInt row, col;

        // We repeat the entire prepartion for "depth" iterations.
        // At very low depths, with only nearest neighbor entanglement, we can avoid entangling the representation of
        // the entire state as a single Schr{\"o}dinger method unit.
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                gateRand = qReg->Rand();

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but "Sqrt" variants are
                // handled as general single bit gates.
                if (gateRand < (ONE_R1 / 3)) {
                    qReg->X(i);
                } else if (gateRand < (2 * ONE_R1 / 3)) {
                    qReg->Y(i);
                } else {
                    qReg->H(i);
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose") the
                // representation without changing the logical state of the QUnit, up to float error:
                qReg->TrySeparate(i);
            }

            std::set<bitLenInt> usedBits;

            for (i = 0; i < (n - rowLen); i++) {
                if (usedBits.find(i) != usedBits.end()) {
                    continue;
                }

                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In the interior bulk, one 2 bit gate is applied for every pair of bits, (as many gates as 1/2 the
                // number of bits). (Unless n is a perfect square, the "row length" has to be factored into a
                // rectangular shape, and "n" is sometimes prime or factors awkwardly.)

                b1 = i;
                if (rowLen == 1U) {
                    b2 = i + 1;
                    usedBits.insert(b1);
                    usedBits.insert(b2);
                } else {
                    // Next row
                    b2 = i + rowLen;

                    if (qReg->Rand() < (ONE_R1 / 2)) {
                        // Next column
                        // (Stop at boundaries of rectangle)
                        if (((i / rowLen) & 1) == 0) {
                            if ((b2 % rowLen) > 0) {
                                b2--;
                            }
                        } else {
                            if ((b2 % rowLen) < (rowLen - 1)) {
                                b2++;
                            }
                        }
                    }

                    usedBits.insert(b1);
                    usedBits.insert(b2);

                    // For the efficiency of QUnit's mapper, we transpose the row and column.
                    col = b1 / rowLen;
                    row = b1 - (col * rowLen);
                    b1 = (row * rowLen) + col;

                    col = b2 / rowLen;
                    row = b2 - (col * rowLen);
                    b2 = (row * rowLen) + col;
                }

                // "iSWAP" is read to be a SWAP operation that imparts a phase factor of i if the bits are
                // different. We use a QUnit SWAP instead, which is very well optimized.
                qReg->Swap(b1, b2);
                // "1/6 of CZ" is read to indicate the 6th root. We use a full CZ instead.
                qReg->CZ(b1, b2);
                // Note that these gates are both symmetric under exchange of "b1" and "b2".
            }
        }

        qReg->MReg(0, n);
    });
}

TEST_CASE("test_quantum_supremacy", "[supreme]")
{
    // This is an attempt to simulate the circuit argued to establish quantum supremacy.

    const int depth = 20;
    benchmarkLoop([](QInterfacePtr qReg, int n) {

        int rowLen = std::sqrt(n);
        while (((n / rowLen) * rowLen) != n) {
            rowLen--;
        }

        complex sixthRoot = std::pow(-ONE_CMPLX, (real1)(1.0 / 6.0));

        real1 gateRand;
        int b1, b2;
        bitLenInt i, d;
        bitLenInt row, col;

        bitLenInt controls[1];

        // We repeat the entire prepartion for "depth" iterations.
        // At very low depths, with only nearest neighbor entanglement, we can avoid entangling the representation of
        // the entire state as a single Schr{\"o}dinger method unit.
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                gateRand = qReg->Rand();

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants are
                // handled as general single bit gates.
                if (gateRand < (ONE_R1 / 3)) {
                    qReg->SqrtX(i);
                } else if (gateRand < (2 * ONE_R1 / 3)) {
                    qReg->SqrtY(i);
                } else {
                    qReg->SqrtH(i);
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose") the
                // representation without changing the logical state of the QUnit, up to float error:
                qReg->TrySeparate(i);
            }

            std::set<bitLenInt> usedBits;

            for (i = 0; i < (n - rowLen); i++) {
                if (usedBits.find(i) != usedBits.end()) {
                    continue;
                }

                // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                // In this test, the boundaries of the rectangle have no couplers.
                // In the interior bulk, one 2 bit gate is applied for every pair of bits, (as many gates as 1/2 the
                // number of bits). (Unless n is a perfect square, the "row length" has to be factored into a
                // rectangular shape, and "n" is sometimes prime or factors awkwardly.)

                b1 = i;
                if (rowLen == 1U) {
                    b2 = i + 1;
                    usedBits.insert(b1);
                    usedBits.insert(b2);
                } else {
                    // Next row
                    b2 = i + rowLen;

                    if (qReg->Rand() < (ONE_R1 / 2)) {
                        // Next column
                        // (Stop at boundaries of rectangle)
                        if (((i / rowLen) & 1) == 0) {
                            if ((b2 % rowLen) > 0) {
                                b2--;
                            }
                        } else {
                            if ((b2 % rowLen) < (rowLen - 1)) {
                                b2++;
                            }
                        }
                    }

                    usedBits.insert(b1);
                    usedBits.insert(b2);

                    // For the efficiency of QUnit's mapper, we transpose the row and column.
                    col = b1 / rowLen;
                    row = b1 - (col * rowLen);
                    b1 = (row * rowLen) + col;

                    col = b2 / rowLen;
                    row = b2 - (col * rowLen);
                    b2 = (row * rowLen) + col;
                }

                // "iSWAP" is read to be a SWAP operation that imparts a phase factor of i if the bits are
                // different.
                qReg->ISwap(b1, b2);
                // "1/6 of CZ" is read to indicate the 6th root.
                controls[0] = b1;
                qReg->ApplyControlledSinglePhase(controls, 1U, b2, ONE_CMPLX, sixthRoot);
                // Note that these gates are both symmetric under exchange of "b1" and "b2".
            }
        }

        qReg->MReg(0, n);
    });
}
