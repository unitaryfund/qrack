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
#include <list>
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

QInterfacePtr MakeRandQubit()
{
    QInterfacePtr qubit = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, ONE_BCI, 0,
        rng, ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng);

    real1 prob = qubit->Rand();
    complex phaseFactor = std::polar(ONE_R1, (real1)(2 * M_PI * qubit->Rand()));

    complex state[2] = { sqrt(ONE_R1 - prob), sqrt(prob) * phaseFactor };
    qubit->SetQuantumState(state);

    return qubit;
}

void benchmarkLoopVariable(std::function<void(QInterfacePtr, int)> fn, bitLenInt mxQbts, bool resetRandomPerm = true,
    bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
{

    const int ITERATIONS = 100;

    std::cout << std::endl;
    std::cout << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
    std::cout << ITERATIONS << " iterations" << std::endl;
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

        if (isBinaryOutput) {
            mOutputFile << std::endl << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
            mOutputFile << ITERATIONS << " iterations" << std::endl;
            mOutputFile << (int)numBits << " qubits" << std::endl;
            mOutputFile << sizeof(bitCapInt) << " bytes in bitCapInt" << std::endl;
        }

        QInterfacePtr qftReg = NULL;
        if (!qUniverse) {
            qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, numBits, 0, rng,
                ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng);
        }
        avgt = 0.0;

        for (i = 0; i < ITERATIONS; i++) {
            if (!qUniverse) {
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
            } else {
                qftReg = MakeRandQubit();
                for (bitLenInt i = 1; i < numBits; i++) {
                    qftReg->Compose(MakeRandQubit());
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

            if (mOutputFileName.compare("")) {
                bitCapInt result = qftReg->MReg(0, numBits);
                if (isBinaryOutput) {
                    mOutputFile.write(reinterpret_cast<char*>(&result), sizeof(bitCapInt));
                } else {
                    mOutputFile << Catch::getResultCapture().getCurrentTestName() << "," << (int)numBits << ","
                                << (uint64_t)result << std::endl;
                }
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
    bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
{
    benchmarkLoopVariable(fn, max_qubits, resetRandomPerm, hadamardRandomBits, logNormal, qUniverse);
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

TEST_CASE("test_quantum_supremacy", "[supreme]")
{
    // This is an attempt to simulate the circuit argued to establish quantum supremacy.
    // See https://doi.org/10.1038/s41586-019-1666-5

    const int depth = 20;

    benchmarkLoop([](QInterfacePtr qReg, int n) {

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        std::list<bitLenInt> gateSequence = { 0, 3, 1, 2, 1, 2, 0, 3 };

        // Depending on which element of the sequential tiling we're running, per depth iteration,
        // we need to start either with row "0" or row "1".
        std::map<bitLenInt, bitLenInt> sequenceRowStart;
        sequenceRowStart[0] = 1;
        sequenceRowStart[1] = 1;
        sequenceRowStart[2] = 0;
        sequenceRowStart[3] = 0;

        // We factor the qubit count into two integers, as close to a perfect square as we can.
        int rowLen = std::sqrt(n);
        while (((n / rowLen) * rowLen) != n) {
            rowLen--;
        }
        int colLen = n / rowLen;

        // "1/6 of a full CZ" is read to indicate the 6th root of the gate operator.
        complex sixthRoot = std::pow(-ONE_CMPLX, (real1)(1.0 / 6.0));

        real1 gateRand;
        bitLenInt gate;
        int b1, b2;
        bitLenInt i, d;
        int row, col;
        int tempRow, tempCol;

        bool startsEvenRow;

        bitLenInt controls[1];

        // We repeat the entire prepartion for "depth" iterations.
        // We can avoid maximal representational entanglement of the state as a single Schr{\"o}dinger method unit.
        // See https://arxiv.org/abs/1710.05867
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
                    // "Square root of W" is understood to be the square root of the Walsh-Hadamard transform,
                    // (a.k.a "H" gate).
                    qReg->SqrtH(i);
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose") the
                // representation without changing the logical state of the QUnit, up to float error:
                // qReg->TrySeparate(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            startsEvenRow = ((sequenceRowStart[gate] & 1U) == 0U);

            for (row = sequenceRowStart[gate]; row < (n / rowLen); row += 2) {
                for (col = 0; col < (n / colLen); col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits, (as
                    // many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length" has to be
                    // factored into a rectangular shape, and "n" is sometimes prime or factors awkwardly.)

                    tempRow = row;
                    tempCol = col;

                    tempRow += ((gate & 2U) ? 1 : -1);

                    if (startsEvenRow) {
                        tempCol += ((gate & 1U) ? 0 : -1);
                    } else {
                        tempCol += ((gate & 1U) ? 1 : 0);
                    }

                    if ((tempRow < 0) || (tempCol < 0) || (tempRow >= rowLen) || (tempCol >= colLen)) {
                        continue;
                    }

                    b1 = row * rowLen + col;
                    b2 = tempRow * rowLen + tempCol;

                    // For the efficiency of QUnit's mapper, we transpose the row and column.
                    tempCol = b1 / rowLen;
                    tempRow = b1 - (tempCol * rowLen);
                    b1 = (tempRow * rowLen) + tempCol;

                    tempCol = b2 / rowLen;
                    tempRow = b2 - (tempCol * rowLen);
                    b2 = (tempRow * rowLen) + tempCol;

                    // "iSWAP" is read to be a SWAP operation that imparts a phase factor of i if the bits are
                    // different.
                    qReg->ISwap(b1, b2);
                    // "1/6 of CZ" is read to indicate the 6th root.
                    controls[0] = b1;
                    qReg->ApplyControlledSinglePhase(controls, 1U, b2, ONE_CMPLX, sixthRoot);
                    // Note that these gates are both symmetric under exchange of "b1" and "b2".
                }
            }
        }

        // We measure all bits once, after the circuit is run.
        qReg->MReg(0, n);
    });
}

TEST_CASE("test_cosmology", "[cosmos]")
{
    // Inspired by https://arxiv.org/abs/1702.06959
    // We assume that the treatment is valid for a bipartite system that has a pure state, entire between interior and
    // (event horizon) boundary degrees of freedom for the Hilbert space. We start with each qubit region subsystem with
    // only internal entanglement between its two internal degrees of freedom, (effectively such that one is interior
    // and the other is boundary, in a totally random basis). We do not explicitly partition between boundary and
    // interior, in part because entanglement can occur internally. We assume the DFT or its inverse is the maximally
    // entangling operation across the ensemble of initially Planck scale separable subsystems. (The finite number of
    // subsystems is only due to resource limit for our model, not any deeper theoretical reason.)

    benchmarkLoop(
        [](QInterfacePtr qUniverse, int n) {
            for (bitLenInt i = 0; i < n; i++) {
                qUniverse->QFT(0, n);
            }
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}
