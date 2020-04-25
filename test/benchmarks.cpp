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
    QInterfacePtr qubit = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 1U, 0, rng,
        ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng);

    real1 prob = qubit->Rand();
    complex phaseFactor = std::polar(ONE_R1, (real1)(2 * M_PI * qubit->Rand()));

    complex state[2] = { ((real1)sqrt(ONE_R1 - prob)) * ONE_CMPLX, ((real1)sqrt(prob)) * phaseFactor };
    qubit->SetQuantumState(state);

    return qubit;
}

void benchmarkLoopVariable(std::function<void(QInterfacePtr, bitLenInt)> fn, bitLenInt mxQbts,
    bool resetRandomPerm = true, bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
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

    QInterfacePtr qftReg = NULL;

    for (numBits = mnQbts; numBits <= mxQbts; numBits++) {

        if (isBinaryOutput) {
            mOutputFile << std::endl << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
            mOutputFile << ITERATIONS << " iterations" << std::endl;
            mOutputFile << (int)numBits << " qubits" << std::endl;
            mOutputFile << sizeof(bitCapInt) << " bytes in bitCapInt" << std::endl;
        }

        if (!qUniverse) {
            if (qftReg != NULL) {
                qftReg.reset();
            }
            qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, numBits, 0, rng,
                ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng, sparse);
        }
        avgt = 0.0;

        for (i = 0; i < ITERATIONS; i++) {
            if (!qUniverse) {
                if (resetRandomPerm) {
                    qftReg->SetPermutation((bitCapIntOcl)(qftReg->Rand() * (bitCapIntOcl)qftReg->GetMaxQPower()));
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
                if (qftReg != NULL) {
                    qftReg.reset();
                }
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

void benchmarkLoop(std::function<void(QInterfacePtr, bitLenInt)> fn, bool resetRandomPerm = true,
    bool hadamardRandomBits = false, bool logNormal = false, bool qUniverse = false)
{
    benchmarkLoopVariable(fn, max_qubits, resetRandomPerm, hadamardRandomBits, logNormal, qUniverse);
}

TEST_CASE("test_cnot_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CNOT(0, 1); });
}

TEST_CASE("test_x_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->X(0); });
}

TEST_CASE("test_y_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Y(0); });
}

TEST_CASE("test_z_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Z(0); });
}

TEST_CASE("test_swap_single", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Swap(0, 1); });
}

TEST_CASE("test_cnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CNOT(0, n / 2, n / 2); });
}

TEST_CASE("test_x_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->X(0, n); });
}

TEST_CASE("test_y_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Y(0, n); });
}

TEST_CASE("test_z_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Z(0, n); });
}

TEST_CASE("test_swap_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->Swap(0, n / 2, n / 2); });
}

TEST_CASE("test_ccnot_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CCNOT(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_and_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->AND(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_or_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->OR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_xor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->XOR(0, n / 3, (2 * n) / 3, n / 3); });
}

TEST_CASE("test_cland_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLAND(0, 0x0c, 0, n); });
}

TEST_CASE("test_clor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_clxor_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CLXOR(0, 0x0d, 0, n); });
}

TEST_CASE("test_rt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->RT(M_PI, 0, n); });
}

TEST_CASE("test_crt_all", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CRT(M_PI, 0, n / 2, n / 2); });
}

TEST_CASE("test_rol", "[gates]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ROL(1, 0, n); });
}

TEST_CASE("test_inc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->INC(1, 0, n); });
}

TEST_CASE("test_incs", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->INCS(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->INCC(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_incsc", "[arithmetic]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->INCSC(1, 0, n - 2, n - 2, n - 1); });
}

TEST_CASE("test_zero_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ZeroPhaseFlip(0, n); });
}

TEST_CASE("test_c_phase_flip_if_less", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->CPhaseFlipIfLess(1, 0, n - 1, n - 1); });
}

TEST_CASE("test_phase_flip", "[phaseflip]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->PhaseFlip(); });
}

TEST_CASE("test_m", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->M(n - 1); });
}

TEST_CASE("test_mreg", "[measure]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->MReg(0, n); });
}

void benchmarkSuperpose(std::function<void(QInterfacePtr, int, unsigned char*)> fn)
{
    bitCapIntOcl i, j;

    bitCapIntOcl wordLength = (max_qubits / 16 + 1);
    bitCapIntOcl indexLength = (1 << (max_qubits / 2));
    unsigned char* testPage = new unsigned char[wordLength * indexLength];
    for (j = 0; j < indexLength; j++) {
        for (i = 0; i < wordLength; i++) {
            testPage[j * wordLength + i] = (j & (0xff << (8 * i))) >> (8 * i);
        }
    }
    benchmarkLoop([fn, testPage](QInterfacePtr qftReg, bitLenInt n) { fn(qftReg, n, testPage); });
    delete[] testPage;
}

TEST_CASE("test_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        qftReg->IndexedLDA(0, n / 2, n / 2, n / 2, testPage);
    });
}

TEST_CASE("test_adc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        qftReg->IndexedADC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_sbc_superposition_reg", "[indexed]")
{
    benchmarkSuperpose([](QInterfacePtr qftReg, bitLenInt n, unsigned char* testPage) {
        qftReg->IndexedSBC(0, (n - 1) / 2, (n - 1) / 2, (n - 1) / 2, (n - 1), testPage);
    });
}

TEST_CASE("test_setbit", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->SetBit(0, true); });
}

TEST_CASE("test_proball", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->ProbAll(0x02); });
}

TEST_CASE("test_set_reg", "[aux]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->SetReg(0, n, 1); });
}

TEST_CASE("test_grover", "[grover]")
{

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 3.

    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        int i;
        // Twelve iterations maximizes the probablity for 256 searched elements, for example.
        // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
        int optIter = M_PI / (4.0 * asin(1.0 / sqrt((real1)pow2(n))));

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
            // Global phase flip has no physically measurable effect:
            // qftReg->PhaseFlip();
        }

        REQUIRE_THAT(qftReg, HasProbability(0x3));

        qftReg->MReg(0, n);
    });
}

TEST_CASE("test_qft_ideal_init", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, false, false);
}

TEST_CASE("test_qft_permutation_init", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, true, false,
        testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_permutation_round_trip_entangled", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, bitLenInt n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_one_way", "[qft]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) { qftReg->QFT(0, n, false); }, true, true,
        testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_qft_superposition_round_trip", "[qft]")
{
    benchmarkLoop(
        [](QInterfacePtr qftReg, bitLenInt n) {
            qftReg->QFT(0, n, false);
            qftReg->IQFT(0, n, false);
        },
        true, true, testEngineType == QINTERFACE_QUNIT);
}

bitLenInt pickRandomBit(QInterfacePtr qReg, std::set<bitLenInt>* unusedBitsPtr)
{
    std::set<bitLenInt>::iterator bitIterator = unusedBitsPtr->begin();
    bitLenInt bitRand = unusedBitsPtr->size() * qReg->Rand();
    if (bitRand >= unusedBitsPtr->size()) {
        bitRand = unusedBitsPtr->size() - 1;
    }
    std::advance(bitIterator, bitRand);
    unusedBitsPtr->erase(bitIterator);
    return *bitIterator;
}

TEST_CASE("test_universal_circuit_continuous", "[supreme]")
{
    const int GateCountMultiQb = 2;
    const int Depth = 20;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {

            int d;
            bitLenInt i;
            real1 theta, phi, lambda;
            bitLenInt b1, b2;
            complex polar0;

            for (d = 0; d < Depth; d++) {

                for (i = 0; i < n; i++) {
                    theta = 2 * M_PI * qReg->Rand();
                    phi = 2 * M_PI * qReg->Rand();
                    lambda = 2 * M_PI * qReg->Rand();

                    qReg->U(i, theta, phi, lambda);
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg, &unusedBits);
                    b2 = pickRandomBit(qReg, &unusedBits);

                    if ((GateCountMultiQb * qReg->Rand()) < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else {
                        qReg->CNOT(b1, b2);
                    }
                }
            }

            qReg->MReg(0, n);
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_discrete", "[supreme]")
{
    const int GateCount1Qb = 2;
    const int GateCountMultiQb = 2;
    const int Depth = 20;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {

            int d;
            bitLenInt i;
            real1 gateRand;
            bitLenInt b1, b2, b3;
            complex polar0;
            int maxGates;

            for (d = 0; d < Depth; d++) {

                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    }
                    // Otherwise, no H gate
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg, &unusedBits);
                    b2 = pickRandomBit(qReg, &unusedBits);

                    if (unusedBits.size() > 0) {
                        maxGates = GateCountMultiQb;
                    } else {
                        maxGates = GateCountMultiQb - 1U;
                    }

                    gateRand = maxGates * qReg->Rand();

                    if (gateRand < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else {
                        b3 = pickRandomBit(qReg, &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }

            qReg->MReg(0, n);
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_digital", "[supreme]")
{
    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 4;
    const int Depth = 20;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {

            int d;
            bitLenInt i;
            real1 gateRand;
            bitLenInt b1, b2, b3;
            complex polar0;
            int maxGates;

            for (d = 0; d < Depth; d++) {

                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        qReg->X(i);
                    } else if (gateRand < (3 * ONE_R1 / GateCount1Qb)) {
                        qReg->Y(i);
                    } else {
                        qReg->T(i);
                    }
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // In the past, "qReg->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
                    // the method can give performance advantages, under opportune conditions, but it does not, here.
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg, &unusedBits);
                    b2 = pickRandomBit(qReg, &unusedBits);

                    if (unusedBits.size() > 0) {
                        maxGates = GateCountMultiQb;
                    } else {
                        maxGates = GateCountMultiQb - 1U;
                    }

                    gateRand = maxGates * qReg->Rand();

                    if (gateRand < ONE_R1) {
                        qReg->Swap(b1, b2);
                    } else if (gateRand < (2 * ONE_R1)) {
                        qReg->CZ(b1, b2);
                    } else if ((unusedBits.size() == 0) || (gateRand < (3 * ONE_R1))) {
                        qReg->CNOT(b1, b2);
                    } else {
                        b3 = pickRandomBit(qReg, &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    }
                }
            }

            qReg->MReg(0, n);
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_universal_circuit_analog", "[supreme]")
{
    const int GateCount1Qb = 3;
    const int GateCountMultiQb = 4;
    const int Depth = 20;

    benchmarkLoop(
        [&](QInterfacePtr qReg, bitLenInt n) {

            int d;
            bitLenInt i;
            real1 gateRand;
            bitLenInt b1, b2, b3;
            bitLenInt control[1];
            complex polar0;
            bool canDo3;
            int gateThreshold, gateMax;

            for (d = 0; d < Depth; d++) {

                for (i = 0; i < n; i++) {
                    gateRand = qReg->Rand();
                    polar0 = std::polar(ONE_R1, (real1)(2 * M_PI * qReg->Rand()));
                    if (gateRand < (ONE_R1 / GateCount1Qb)) {
                        qReg->H(i);
                    } else if (gateRand < (2 * ONE_R1 / GateCount1Qb)) {
                        qReg->ApplySinglePhase(ONE_CMPLX, polar0, i);
                    } else {
                        qReg->ApplySingleInvert(ONE_CMPLX, polar0, i);
                    }
                }

                std::set<bitLenInt> unusedBits;
                for (i = 0; i < n; i++) {
                    // TrySeparate hurts average time, in this case, but it majorly benefits statistically common
                    // worse cases, on these random circuits.
                    qReg->TrySeparate(i);
                    unusedBits.insert(unusedBits.end(), i);
                }

                while (unusedBits.size() > 1) {
                    b1 = pickRandomBit(qReg, &unusedBits);
                    b2 = pickRandomBit(qReg, &unusedBits);

                    canDo3 = (unusedBits.size() > 0);
                    if (canDo3) {
                        gateThreshold = 3;
                        gateMax = GateCountMultiQb;
                    } else {
                        gateThreshold = 2;
                        gateMax = GateCountMultiQb - 1;
                    }

                    gateRand = qReg->Rand();
                    if (gateRand < (ONE_R1 / gateMax)) {
                        qReg->Swap(b1, b2);
                    } else if (canDo3 && (gateRand < (2 * ONE_R1 / GateCountMultiQb))) {
                        b3 = pickRandomBit(qReg, &unusedBits);
                        qReg->CCNOT(b1, b2, b3);
                    } else {
                        control[0] = b1;
                        polar0 = std::polar(ONE_R1, (real1)(2 * M_PI * qReg->Rand()));
                        if (gateRand < (gateThreshold * ONE_R1 / gateMax)) {
                            qReg->ApplyControlledSinglePhase(control, 1U, b2, polar0, -polar0);
                        } else {
                            qReg->ApplyControlledSingleInvert(control, 1U, b2, polar0, polar0);
                        }
                    }
                }
            }

            qReg->MReg(0, n);
        },
        false, false, testEngineType == QINTERFACE_QUNIT);
}

TEST_CASE("test_quantum_supremacy", "[supreme]")
{
    // This is an attempt to simulate the circuit argued to establish quantum supremacy.
    // See https://doi.org/10.1038/s41586-019-1666-5

    const int depth = 20;

    benchmarkLoop([&](QInterfacePtr qReg, bitLenInt n) {

        // The test runs 2 bit gates according to a tiling sequence.
        // The 1 bit indicates +/- column offset.
        // The 2 bit indicates +/- row offset.
        // This is the "ABCDCDAB" pattern, from the Cirq definition of the circuit in the supplemental materials to the
        // paper.
        std::list<bitLenInt> gateSequence = { 0, 3, 2, 1, 2, 1, 0, 3 };

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

        std::vector<int> lastSingleBitGates;
        std::set<int>::iterator gateChoiceIterator;
        int gateChoice;

        // We repeat the entire prepartion for "depth" iterations.
        // We can avoid maximal representational entanglement of the state as a single Schr{\"o}dinger method unit.
        // See https://arxiv.org/abs/1710.05867
        for (d = 0; d < depth; d++) {
            for (i = 0; i < n; i++) {
                gateRand = qReg->Rand();

                // Each individual bit has one of these 3 gates applied at random.
                // Qrack has optimizations for gates including X, Y, and particularly H, but these "Sqrt" variants
                // are handled as general single bit gates.

                // The same gate is not applied twice consecutively in sequence.

                if (d == 0) {
                    // For the first iteration, we can pick any gate.

                    if (gateRand < (ONE_R1 / 3)) {
                        qReg->SqrtX(i);
                        lastSingleBitGates.push_back(0);
                    } else if (gateRand < (2 * ONE_R1 / 3)) {
                        qReg->SqrtY(i);
                        lastSingleBitGates.push_back(1);
                    } else {
                        // "Square root of W" appears to be equivalent to T.SqrtX.IT, looking at the definition in the
                        // supplemental materials.
                        qReg->SqrtXConjT(i);
                        lastSingleBitGates.push_back(2);
                    }
                } else {
                    // For all subsequent iterations after the first, we eliminate the choice of the same gate applied
                    // on the immediately previous iteration.

                    std::set<int> gateChoices = { 0, 1, 2 };
                    gateChoiceIterator = gateChoices.begin();
                    std::advance(gateChoiceIterator, lastSingleBitGates[i]);
                    gateChoices.erase(gateChoiceIterator);

                    gateChoiceIterator = gateChoices.begin();
                    std::advance(gateChoiceIterator, (gateRand < (ONE_R1 / 2)) ? 0 : 1);
                    gateChoices.erase(gateChoiceIterator);

                    gateChoice = *(gateChoices.begin());

                    if (gateChoice == 0) {
                        qReg->SqrtX(i);
                        lastSingleBitGates[i] = 0;
                    } else if (gateChoice == 1) {
                        qReg->SqrtY(i);
                        lastSingleBitGates[i] = 1;
                    } else {
                        // "Square root of W" appears to be equivalent to T.SqrtX.IT, looking at the definition in the
                        // supplemental materials.
                        qReg->SqrtXConjT(i);
                        lastSingleBitGates[i] = 2;
                    }
                }

                // This is a QUnit specific optimization attempt method that can "compress" (or "Schmidt decompose")
                // the representation without changing the logical state of the QUnit, up to float error:
                // qReg->TrySeparate(i);
            }

            gate = gateSequence.front();
            gateSequence.pop_front();
            gateSequence.push_back(gate);

            startsEvenRow = ((sequenceRowStart[gate] & 1U) == 0U);

            for (row = sequenceRowStart[gate]; row < (int)(n / rowLen); row += 2) {
                for (col = 0; col < (int)(n / colLen); col++) {
                    // The following pattern is isomorphic to a 45 degree bias on a rectangle, for couplers.
                    // In this test, the boundaries of the rectangle have no couplers.
                    // In a perfect square, in the interior bulk, one 2 bit gate is applied for every pair of bits,
                    // (as many gates as 1/2 the number of bits). (Unless n is a perfect square, the "row length"
                    // has to be factored into a rectangular shape, and "n" is sometimes prime or factors
                    // awkwardly.)

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
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // We assume that the treatment of that work is valid for a bipartite system that has a pure state, entire
    // between interior and (event horizon) boundary degrees of freedom for the Hilbert space. We start with each
    // qubit region subsystem with only internal entanglement between its two internal degrees of freedom,
    // (effectively such that one is interior and the other is boundary, in a totally random basis). We do not
    // explicitly partition between boundary and interior, in part because entanglement can occur internally. We
    // assume the DFT or its inverse is the maximally entangling operation across the ensemble of initially Planck
    // scale separable subsystems. The finite number of subsystems is due to resource limit for our model, but it
    // might effectively represent an entanglement or "entropy" budget for a closed universe; the time to maximum
    // entanglement for "n" available qubits should be "n" Planck time steps on average. (The von Neumann entropy
    // actually remains 0, in this entire simulation, as the state is pure and evolves in a unitary fashion, but, if
    // unitary evolution holds for the entire real physical cosmological system of our universe, then this
    // entangling action gives rise to the appearance of non-zero von Neumann entropy of a mixed state.)  We limit
    // to the 1 spatial + 1 time dimension case.
    //
    // If the (inverse) DFT is truly maximally entangling, it might not be appropriate to iterate the full-width,
    // monotonic DFT as a time-step, because this then consumes the entire entropy budget of the Hubble sphere in
    // one step. Further, deterministic progression toward higher entanglement, and therefore higher effective
    // entropy, assumes a fixed direction for the "arrow of time." Given the time symmetry of unitary evolution,
    // hopefully, the thermodynamic arrow of time would be emergent in a very-early-universe model, rather than
    // assumed to be fixed. As such, suppose that there is locally a 0.5/0.5 of 1.0 probability for either direction
    // of apparent time in a step, represented by randomly choosing QFT or inverse on a local region. Further,
    // initially indepedent regions cannot be causally influenced by distant regions faster than the speed of light,
    // where the light cone grows at a rate of one Planck distance per Planck time. Locality implies that, in one
    // Planck time step, a 2 qubit (inverse) DFT can be acted between each nearest-neighbor pair. We also assume
    // that causally disconnected regions develop local entanglement in parallel. However, if we took a longer time
    // step, an integer multiple of the Planck time, then higher order QFTs would be needed to simulate the step.
    // Probably, the most accurate simulation would take the "squarest" possible time step by space step, but then
    // this is simply a single QFT or its inverse for the entire entropy budget of the space. (We must acknowledge,
    // it is apparent to us that this simulation we choose is a problem that can be made relatively easy for
    // Qrack::QUnit.)

    // "RandInit" -
    // true - initialize all qubits with completely random (single qubit, separable) states
    // false - initialize entire register as |0>
    //
    // Setting a totally random eigenstate for each bit simulates the limits of causality, since qubits have not had
    // time to interact with each other and reach homogeneity. However, if the initial state of each region is an
    // eigenstate, then maybe we can call each initial state the local |0> state, by convention. (This might not
    // actually be self-consistent; the limitation on causality and homogeneity might preempt the validity of this
    // initialization. It might still be an interesting case to consider, and to debug with.)
    const bool RandInit = true;

    // "UseTDepth"
    // true - for "n" qubits, simulate time to depth "n"
    // false - simulate to "depth" time steps
    const bool UseTDepth = true;
    const int TDepth = 8;
    // Time step of simulation, (in "Planck times")
    const bitLenInt TStep = 1;
    // If true, loop the parallel local evolution back around on the boundaries of the qubit array.
    const bool DoOrbifold = true;

    benchmarkLoop(
        [&](QInterfacePtr qUniverse, bitLenInt n) {
            int t, x;
            int tMax = UseTDepth ? TDepth : n;

            for (t = 1; t < tMax; t += TStep) {
                for (x = 0; x < (int)(n - TStep); x++) {
                    if (qUniverse->Rand() < (ONE_R1 / 2)) {
                        qUniverse->QFT(x, TStep + 1U);
                    } else {
                        qUniverse->IQFT(x, TStep + 1U);
                    }
                }

                if (!DoOrbifold) {
                    continue;
                }

                // Orbifold the last and first bits.
                qUniverse->ROL(TStep, 0, n);
                for (x = 0; x < (int)TStep; x++) {
                    if (qUniverse->Rand() < (ONE_R1 / 2)) {
                        qUniverse->QFT(x, TStep + 1U);
                    } else {
                        qUniverse->IQFT(x, TStep + 1U);
                    }
                }
                qUniverse->ROR(TStep, 0, n);
            }
        },
        false, false, false, RandInit);
}

TEST_CASE("test_qft_cosmology", "[cosmos]")
{
    // This is "scratch work" inspired by https://arxiv.org/abs/1702.06959
    //
    // Per the notes in the previous test, this is probably our most accurate possible simulation of a cosmos: one
    // QFT (or inverse) to consume the entire "entropy" budget.
    //
    // Note that, when choosing between QFT and inverse QFT, AKA inverse DFT and DFT respectively, the choice of the
    // QFT over the IQFT is not entirely arbitrary: we are mapping from a single phase in the phase space of
    // potential universes to a single configuration. Remember that we initialize as a collection of entirely
    // random, single, separable qubits.

    benchmarkLoop([&](QInterfacePtr qUniverse, bitLenInt n) { qUniverse->QFT(0, n); }, false, false, false, true);
}

TEST_CASE("test_n_bell", "[stabilizer]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        qftReg->H(0);
        for (bitLenInt i = 0; i < (n - 1); i++) {
            qftReg->CNOT(i, i + 1U);
        }
    });
}

TEST_CASE("test_repeat_h_cnot", "[stabilizer]")
{
    benchmarkLoop([](QInterfacePtr qftReg, bitLenInt n) {
        for (bitLenInt i = 0; i < (n - 1); i++) {
            qftReg->H(i);
            qftReg->CNOT(i, i + 1U);
        }
    });
}

struct MultiQubitGate {
    int gate;
    bitLenInt b1;
    bitLenInt b2;
    bitLenInt b3;
};

TEST_CASE("test_universal_circuit_digital_cross_entropy", "[supreme]")
{
    std::cout << ">>> 'test_universal_circuit_digital_cross_entropy':" << std::endl;

    const int GateCount1Qb = 4;
    const int GateCountMultiQb = 4;
    const int Depth = 3;

    const int ITERATIONS = 20000;
    const int n = 8;
    bitCapInt permCount = pow2(n);
    bitCapInt perm;

    std::cout << "Width: " << n << " qubits" << std::endl;
    std::cout << "Depth: " << Depth << " layers of 1 qubit then multi-qubit gates" << std::endl;
    std::cout << "samples collected: " << ITERATIONS << std::endl;

    int d;
    bitLenInt i;
    std::vector<std::vector<int>> gate1QbRands(Depth);
    std::vector<std::vector<MultiQubitGate>> gateMultiQbRands(Depth);
    complex polar0;
    int maxGates;

    QInterfacePtr goldStandard = CreateQuantumInterface(testSubSubEngineType, n, 0, rng, ONE_CMPLX,
        enable_normalization, true, false, device_id, !disable_hardware_rng);

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < n; i++) {
            layer1QbRands.push_back((int)(goldStandard->Rand() * GateCount1Qb));
        }

        std::set<bitLenInt> unusedBits;
        for (i = 0; i < n; i++) {
            // In the past, "goldStandard->TrySeparate(i)" was also used, here, to attempt optimization. Be aware that
            // the method can give performance advantages, under opportune conditions, but it does not, here.
            unusedBits.insert(unusedBits.end(), i);
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        while (unusedBits.size() > 1) {
            MultiQubitGate multiGate;
            multiGate.b1 = pickRandomBit(goldStandard, &unusedBits);
            multiGate.b2 = pickRandomBit(goldStandard, &unusedBits);

            if (unusedBits.size() > 0) {
                maxGates = GateCountMultiQb;
            } else {
                maxGates = GateCountMultiQb - 1U;
            }

            multiGate.gate = maxGates * goldStandard->Rand();

            if (multiGate.gate > 2) {
                multiGate.b3 = pickRandomBit(goldStandard, &unusedBits);
            }

            layerMultiQbRands.push_back(multiGate);
        }
    }

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < layer1QbRands.size(); i++) {
            int gate1Qb = layer1QbRands[i];
            if (gate1Qb == 0) {
                goldStandard->H(i);
            } else if (gate1Qb == 1) {
                goldStandard->X(i);
            } else if (gate1Qb == 2) {
                goldStandard->Y(i);
            } else {
                goldStandard->T(i);
            }
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                goldStandard->Swap(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 1) {
                goldStandard->CZ(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 2) {
                goldStandard->CNOT(multiGate.b1, multiGate.b2);
            } else {
                goldStandard->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
            }
        }
    }

    bitCapInt qPowers[n];
    for (i = 0; i < n; i++) {
        qPowers[i] = pow2(i);
    }

    std::map<bitCapInt, int> goldStandardResult = goldStandard->MultiShotMeasureMask(qPowers, n, ITERATIONS);

    std::map<bitCapInt, int>::iterator measurementBin;

    real1 uniformRandomCount = ITERATIONS / (real1)permCount;
    int goldBinResult;
    real1 crossEntropy = ZERO_R1;
    for (perm = 0; perm < permCount; perm++) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }
        crossEntropy += (uniformRandomCount - goldBinResult) * (uniformRandomCount - goldBinResult);
    }
    if (crossEntropy < ZERO_R1) {
        crossEntropy = ZERO_R1;
    }
    crossEntropy = ONE_R1 - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. uniform random cross entropy (out of 1.0): " << crossEntropy << std::endl;

    std::map<bitCapInt, int> goldStandardResult2 = goldStandard->MultiShotMeasureMask(qPowers, n, ITERATIONS);

    int testBinResult;
    crossEntropy = ZERO_R1;
    for (perm = 0; perm < permCount; perm++) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = goldStandardResult2.find(perm);
        if (measurementBin == goldStandardResult2.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1) {
        crossEntropy = ZERO_R1;
    }
    crossEntropy = ONE_R1 - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. gold standard cross entropy (out of 1.0): " << crossEntropy << std::endl;

    QInterfacePtr testCase = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, n, 0, rng,
        ONE_CMPLX, enable_normalization, true, false, device_id, !disable_hardware_rng, sparse);

    std::map<bitCapInt, int> testCaseResult;

    //for (int iter = 0; iter < ITERATIONS; iter++) {
        testCase->SetPermutation(0);
        for (d = 0; d < Depth; d++) {
            std::vector<int>& layer1QbRands = gate1QbRands[d];
            for (i = 0; i < layer1QbRands.size(); i++) {
                int gate1Qb = layer1QbRands[i];
                if (gate1Qb == 0) {
                    testCase->H(i);
                } else if (gate1Qb == 1) {
                    testCase->X(i);
                } else if (gate1Qb == 2) {
                    testCase->Y(i);
                } else {
                    testCase->T(i);
                }
            }

            std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
            for (i = 0; i < layerMultiQbRands.size(); i++) {
                MultiQubitGate multiGate = layerMultiQbRands[i];
                if (multiGate.gate == 0) {
                    testCase->Swap(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 1) {
                    testCase->CZ(multiGate.b1, multiGate.b2);
                } else if (multiGate.gate == 2) {
                    testCase->CNOT(multiGate.b1, multiGate.b2);
                } else {
                    testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
                }
            }
        }

    //    perm = testCase->MReg(0, n);
    //    if (testCaseResult.find(perm) == testCaseResult.end()) {
    //        testCaseResult[perm] = 1;
    //    } else {
    //        testCaseResult[perm] += 1;
    //    }
    //}
    // Comment out the ITERATIONS loop and testCaseResult[perm] update above, and uncomment this line below, for a
    // faster benchmark. This will not test the effect of the MReg() method.
    testCaseResult = testCase->MultiShotMeasureMask(qPowers, n, ITERATIONS);

    crossEntropy = ZERO_R1;
    for (perm = 0; perm < permCount; perm++) {
        measurementBin = goldStandardResult.find(perm);
        if (measurementBin == goldStandardResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = testCaseResult.find(perm);
        if (measurementBin == testCaseResult.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1) {
        crossEntropy = ZERO_R1;
    }
    crossEntropy = ONE_R1 - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Gold standard vs. test case cross entropy (out of 1.0): " << crossEntropy << std::endl;

    std::map<bitCapInt, int> testCaseResult2;

    testCase->SetPermutation(0);

    for (d = 0; d < Depth; d++) {
        std::vector<int>& layer1QbRands = gate1QbRands[d];
        for (i = 0; i < layer1QbRands.size(); i++) {
            int gate1Qb = layer1QbRands[i];
            if (gate1Qb == 0) {
                testCase->H(i);
            } else if (gate1Qb == 1) {
                testCase->X(i);
            } else if (gate1Qb == 2) {
                testCase->Y(i);
            } else {
                testCase->T(i);
            }
        }

        std::vector<MultiQubitGate>& layerMultiQbRands = gateMultiQbRands[d];
        for (i = 0; i < layerMultiQbRands.size(); i++) {
            MultiQubitGate multiGate = layerMultiQbRands[i];
            if (multiGate.gate == 0) {
                testCase->Swap(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 1) {
                testCase->CZ(multiGate.b1, multiGate.b2);
            } else if (multiGate.gate == 2) {
                testCase->CNOT(multiGate.b1, multiGate.b2);
            } else {
                testCase->CCNOT(multiGate.b1, multiGate.b2, multiGate.b3);
            }
        }
    }
    testCaseResult2 = testCase->MultiShotMeasureMask(qPowers, n, ITERATIONS);

    crossEntropy = ZERO_R1;
    for (perm = 0; perm < permCount; perm++) {
        measurementBin = testCaseResult.find(perm);
        if (measurementBin == testCaseResult.end()) {
            goldBinResult = 0;
        } else {
            goldBinResult = measurementBin->second;
        }

        measurementBin = testCaseResult2.find(perm);
        if (measurementBin == testCaseResult2.end()) {
            testBinResult = 0;
        } else {
            testBinResult = measurementBin->second;
        }
        crossEntropy += (testBinResult - goldBinResult) * (testBinResult - goldBinResult);
    }
    if (crossEntropy < ZERO_R1) {
        crossEntropy = ZERO_R1;
    }
    crossEntropy = ONE_R1 - sqrt(crossEntropy) / ITERATIONS;
    std::cout << "Test case vs. (duplicate) test case cross entropy (out of 1.0): " << crossEntropy << std::endl;
}
