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

#include "tests.hpp"

using namespace Qrack;

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B) do {                                            \
        double __tmp_a = A;                                                 \
        double __tmp_b = B;                                                 \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                             \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                             \
    } while (0);

void benchmarkLoop(std::function<void(QInterfacePtr, int)> fn) {

    const int ITERATIONS = 100;

    std::cout<<std::endl;
    std::cout<<ITERATIONS<<" iterations";
    std::cout<<std::endl;
    std::cout<<"# of Qubits, ";
    std::cout<<"Average Time (ms), ";
    std::cout<<"Std. Error (ms)"<<std::endl;

    clock_t startClock, tClock, allClock;
    clock_t trialClocks[ITERATIONS];

    int i, numBits;

    double avgt, stdet;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    for (numBits = 3; numBits <= 20; numBits++) {
        QInterfacePtr qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, numBits, 0, rng);
        startClock = clock();
        allClock = startClock;
        tClock = startClock;
        avgt = 0.0;
        for (i = 0; i < ITERATIONS; i++) {

            // Run loop body
            fn(qftReg, numBits);

            // Display timing
            tClock = clock() - allClock;
            trialClocks[i] = tClock;
            avgt += tClock;
            allClock = clock();
        }
        allClock = clock() - startClock;
        avgt /= ITERATIONS;

        stdet = 0.0;
        for (i = 0; i < ITERATIONS; i++) {
            stdet += (trialClocks[i] - avgt) * (trialClocks[i] - avgt);
        }
        stdet = sqrt(stdet / ITERATIONS);

        std::cout<<(int)numBits<<", ";
        std::cout<<(avgt * 1000.0 / CLOCKS_PER_SEC)<<", ";
        std::cout<<(stdet * 1000.0 / CLOCKS_PER_SEC)<<std::endl;
    }
}

TEST_CASE("test_cnot")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CNOT(0, n/2, n/2);});
}

TEST_CASE("test_anticnot")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->AntiCNOT(0, n/2, n/2);});
}

TEST_CASE("test_ccnot")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CCNOT(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE("test_anticcnot")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->AntiCCNOT(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE("test_swap")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->Swap(0, n/2, n/2);});
}

TEST_CASE("test_x")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->X(0, n);});
}

TEST_CASE("test_y")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->Y(0, n);});
}

TEST_CASE("test_z")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->Z(0, n);});
}

TEST_CASE("test_cy")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CY(0, n/2, n/2);});
}

TEST_CASE("test_cz")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CZ(0, n/2, n/2);});
}

TEST_CASE("test_and")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->AND(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE("test_or")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->OR(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE("test_xor")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->XOR(0, n/3, (2*n)/3, n/3);});
}

TEST_CASE("test_cland")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CLAND(0, 0x0c, 0, n);});
}

TEST_CASE("test_clor")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CLOR(0, 0x0d, 0, n);});
}

TEST_CASE("test_clxor")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CLXOR(0, 0x0d, 0, n);});
}

TEST_CASE("test_rt_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RT(M_PI, 0, n);});
}

TEST_CASE("test_rtdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RTDyad(1, 1, 0, n);});
}

TEST_CASE("test_crt_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRT(M_PI, 0, n/2, n/2);});
}

TEST_CASE("test_crtdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRTDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE("test_rx_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RX(M_PI, 0, n);});
}

TEST_CASE("test_rxdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RXDyad(1, 1, 0, n);});
}

TEST_CASE("test_crx_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRX(M_PI, 0, n/2, n/2);});
}

TEST_CASE("test_crxdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRXDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE("test_ry_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RY(M_PI, 0, n);});
}

TEST_CASE("test_rydyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RYDyad(1, 1, 0, n/2);});
}

TEST_CASE("test_cry_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRY(M_PI, 0, n/2, n/2);});
}

TEST_CASE("test_crydyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRYDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE("test_rz_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RZ(M_PI, 0, n);});
}

TEST_CASE("test_rzdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->RZDyad(1, 1, 0, n);});
}

TEST_CASE("test_crz_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRZ(M_PI, 0, n/2, n/2);});
}

TEST_CASE("test_crzdyad_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CRZDyad(1, 1, 0, n/2, n/2);});
}

TEST_CASE("test_rol")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ROL(1, 0, n);});
}

TEST_CASE("test_ror")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ROR(1, 0, n);});
}

TEST_CASE("test_asl")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ASL(1, 0, n);});
}

TEST_CASE("test_asr")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ASR(1, 0, n);});
}

TEST_CASE("test_lsl")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->LSL(1, 0, n);});
}

TEST_CASE("test_lsr")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->LSR(1, 0, n);});
}

TEST_CASE("test_inc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INC(1, 0, n);});
}

TEST_CASE("test_incs")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INCS(1, 0, n-1, n-1);});
}

TEST_CASE("test_incc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INCC(1, 0, n-1, n-1);});
}

/*
TEST_CASE("test_incbcd")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INCBCD(1, 0, n-1);});
}

TEST_CASE("test_incbcdc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INCBCDC(1, 0, n-1, n-1);});
}
*/

TEST_CASE("test_incsc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->INCSC(1, 0, n-2, n-2, n-1);});
}

TEST_CASE("test_dec")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DEC(1, 0, n);});
}

TEST_CASE("test_decs")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DECS(1, 0, n-1, n-1);});
}

TEST_CASE("test_decc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DECC(1, 0, n-1, n-1);});
}

/*
TEST_CASE("test_decbcd")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DECBCD(1, 0, n-1);});
}

TEST_CASE("test_decbcdc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DECBCDC(1, 0, n-1, n-1);});
}
*/

TEST_CASE("test_decsc")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->DECSC(1, 0, n-2, n-2, n-1);});
}

TEST_CASE("test_qft_h")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->QFT(0, n);});
}

TEST_CASE("test_zero_phase_flip")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ZeroPhaseFlip(0, n);});
}


TEST_CASE("test_c_phase_flip_if_less")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->CPhaseFlipIfLess(1, 0, n - 1, n - 1);});
}

TEST_CASE("test_phase_flip")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->PhaseFlip();});
}

TEST_CASE("test_m")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->M(n-1);});
}

TEST_CASE("test_mreg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->MReg(0, n);});
}


TEST_CASE("test_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->IndexedLDA(0, n/2, n/2, n/2, testPage);});
}

TEST_CASE("test_adc_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->IndexedADC(0, (n-1)/2, (n-1)/2, (n-1)/2, (n-1), testPage);});
}

TEST_CASE("test_sbc_superposition_reg")
{
    int j;

    unsigned char testPage[2048];
    for (j = 0; j < 1024; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->IndexedSBC(0, (n-1)/2, (n-1)/2, (n-1)/2, (n-1), testPage);});
}

TEST_CASE("test_setbit")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->SetBit(0, true);});
}

TEST_CASE("test_proball")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->ProbAll(0x02);});
}

TEST_CASE("test_set_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->SetReg(0, n, 1);});
}

TEST_CASE("test_swap_reg")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){qftReg->Swap(0, n/2, n/2);});
}

#if 0
TEST_CASE("test_grover")
{
    benchmarkLoop([&](QInterfacePtr qftReg, int n){
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
