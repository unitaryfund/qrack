//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <atomic>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "qfactory.hpp"

#include "benchmarks.hpp"

using namespace Qrack;

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B) do {                                            \
        double __tmp_a = A;                                                 \
        double __tmp_b = B;                                                 \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                             \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                             \
    } while (0);

void benchmarkLoop(std::shared_ptr<std::default_random_engine> rng, QInterfaceEngine engineType, QInterfaceEngine subEngineType,std::function<void(QInterfacePtr, int)> fn) {

    std::cout<<std::endl;
    std::cout<<"All Time, ";
    std::cout<<"Average, ";
    std::cout<<"Std. Error"<<std::endl;

    const int ITERATIONS = 100;

    clock_t start, t, all;
    clock_t trials[ITERATIONS];

    int i, j;

    double avgt, stdet;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    for (j = 3; j <= 20; j++) {
        QInterfacePtr qftReg = CreateQuantumInterface(engineType, subEngineType, j, 0, rng);
        start = clock();
        all = start;
        t = start;
        avgt = 0.0;
        for (i = 0; i < ITERATIONS; i++) {

            // Run loop body
            fn(qftReg, j);

            // Display timing
            t = clock() - all;
            trials[i] = t;
            avgt += t;
            all = clock();
        }
        all = clock() - start;
        avgt /= ITERATIONS;

        stdet = 0.0;
        for (i = 0; i < ITERATIONS; i++) {
            stdet += (trials[i] - avgt) * (trials[i] - avgt);
        }
        stdet = sqrt(stdet / ITERATIONS);

        std::cout<<(all * 1000.0 / CLOCKS_PER_SEC)<<", ";
        std::cout<<(avgt * 1000.0 / CLOCKS_PER_SEC)<<", ";
        std::cout<<(stdet * 1000.0 / CLOCKS_PER_SEC)<<std::endl;
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cnot")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CNOT(0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticnot")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->AntiCNOT(0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccnot")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CCNOT(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcnot")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->AntiCCNOT(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_not")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->X(0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->Swap(0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_x")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->X(0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_y")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->Y(0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_z")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->Z(0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cy")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CY(0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cz")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CZ(0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_and")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->AND(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_or")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->OR(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xor")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->XOR(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cland")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CLAND(0, 0x0c, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_clor")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CLOR(0, 0x0d, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_clxor")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CLXOR(0, 0x0d, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rt_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RT(M_PI, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rtdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RTDyad(1, 1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crt_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRT(M_PI, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crtdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRTDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rx_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RX(M_PI, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rxdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RXDyad(1, 1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crx_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRX(M_PI, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crxdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRXDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RY(M_PI, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rydyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RYDyad(1, 1, 0, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cry_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRY(M_PI, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crydyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRYDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rz_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RZ(M_PI, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rzdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->RZDyad(1, 1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crz_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRZ(M_PI, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crzdyad_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CRZDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rol")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ROL(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ror")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ROR(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_asl")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ASL(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_asr")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ASR(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsl")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->LSL(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsr")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->LSR(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_inc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INC(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incs")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INCS(1, 0, n-1, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INCC(1, 0, n-1, n-1);});
}

/*
TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcd")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INCBCD(1, 0, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcdc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INCBCDC(1, 0, n-1, n-1);});
}
*/

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incsc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->INCSC(1, 0, n-2, n-2, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dec")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DEC(1, 0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decs")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DECS(1, 0, n-1, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DECC(1, 0, n-1, n-1);});
}

/*
TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcd")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DECBCD(1, 0, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcdc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DECBCDC(1, 0, n-1, n-1);});
}
*/

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decsc")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->DECSC(1, 0, n-2, n-2, n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qft_h")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->QFT(0, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_zero_phase_flip")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ZeroPhaseFlip(0, n);});
}


TEST_CASE_METHOD(QInterfaceTestFixture, "test_c_phase_flip_if_less")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->CPhaseFlipIfLess(1, 0, n, n);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_phase_flip")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->PhaseFlip();});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->M(n-1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mreg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->MReg(0, n);});
}


TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->IndexedLDA(0, n/2, n/2, n/2, testPage);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->IndexedADC(0, (n-1)/2, (n-1)/2, (n-1)/2, (n-1), testPage);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->IndexedSBC(0, (n-1)/2, (n-1)/2, (n-1)/2, (n-1), testPage);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_setbit")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->SetBit(0, true);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_proball")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->ProbAll(0x02);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_set_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->SetReg(0, n, 1);});
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_reg")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){qftReg->Swap(0, n/2, n/2);});
}

#if 0
TEST_CASE_METHOD(QInterfaceTestFixture, "test_grover")
{
    benchmarkLoop(rng, engineType, subEngineType, [&](QInterfacePtr qftReg, int n){
        qftReg->SetPermutation(0);
        qftReg->H(0, 8);

        // std::cout << "Iterations:" << std::endl;
        // Twelve iterations maximizes the probablity for 256 searched elements.
        for (i = 0; i < 12; i++) {
            qftReg->DEC(100, 0, 8);
            qftReg->ZeroPhaseFlip(0, 8);
            qftReg->INC(100, 0, 8);
            // This ends the "oracle."
            qftReg->H(0, 8);
            qftReg->ZeroPhaseFlip(0, 8);
            qftReg->H(0, 8);
            qftReg->PhaseFlip();
        }
    });
}
#endif
