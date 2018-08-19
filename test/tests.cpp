//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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

#define ALIGN_SIZE 64
#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1 __tmp_a = A;                                                                                             \
        real1 __tmp_b = B;                                                                                             \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                                                                        \
    } while (0);

void print_bin(int bits, int d);
void log(QInterfacePtr p);

void print_bin(int bits, int d)
{
    int mask = 1 << bits;
    while (mask != 0) {
        printf("%d", !!(d & mask));
        mask >>= 1;
    }
}

void log(QInterfacePtr p) { std::cout << std::endl << std::showpoint << p << std::endl; }

unsigned char* cl_alloc(size_t ucharCount)
{
#ifdef __APPLE__
    void* toRet;
    posix_memalign(&toRet, ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
    return (unsigned char*)toRet;
#else
    return (unsigned char*)aligned_alloc(ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
#endif
}

TEST_CASE("test_complex")
{
    bool test;
    complex cmplx1(1.0, -1.0);
    complex cmplx2(-0.5, 0.5);
    complex cmplx3(0.0, 0.0);

    REQUIRE(cmplx1 != cmplx2);

    REQUIRE(conj(cmplx1) == complex(1.0, 1.0));

    test = ((real1)abs(cmplx1) > (real1)(sqrt(2.0) - EPSILON)) && ((real1)abs(cmplx1) < (real1)(sqrt(2.0) + EPSILON));
    REQUIRE(test);

    cmplx3 = polar(1.0, M_PI / 2.0);
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 + cmplx2;
    test = (real(cmplx3) > (real1)(0.5 - EPSILON)) && (real(cmplx3) < (real1)(0.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-0.5 - EPSILON)) && (imag(cmplx3) < (real1)(-0.5 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 - cmplx2;
    test = (real(cmplx3) > (real1)(1.5 - EPSILON)) && (real(cmplx3) < (real1)(1.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-1.5 - EPSILON)) && (imag(cmplx3) < (real1)(-1.5 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 * cmplx2;
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1;
    cmplx3 *= cmplx2;
    test = (real(cmplx3) > (real1)(0.0 - EPSILON)) && (real(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) && (imag(cmplx3) < (real1)(1.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx1 / cmplx2;
    test = (real(cmplx3) > (real1)(-2.0 - EPSILON)) && (real(cmplx3) < (real1)(-2.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) && (imag(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = cmplx2;
    cmplx3 /= cmplx1;
    test = (real(cmplx3) > (real1)(-0.5 - EPSILON)) && (real(cmplx3) < (real1)(-0.5 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) && (imag(cmplx3) < (real1)(0.0 + EPSILON));
    REQUIRE(test);

    cmplx3 = ((real1)2.0) * cmplx1;
    test = (real(cmplx3) > (real1)(2.0 - EPSILON)) && (real(cmplx3) < (real1)(2.0 + EPSILON));
    REQUIRE(test);
    test = (imag(cmplx3) > (real1)(-2.0 - EPSILON)) && (imag(cmplx3) < (real1)(-2.0 + EPSILON));
    REQUIRE(test);
}

TEST_CASE("test_qengine_cpu_par_for")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    int NUM_ENTRIES = 2000;
    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for(0, NUM_ENTRIES, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
        calls++;
    });

    REQUIRE(calls.load() == NUM_ENTRIES);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        REQUIRE(hit[i].load() == true);
    }
}

TEST_CASE("test_qengine_cpu_par_for_skip")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    int NUM_ENTRIES = 2000;
    int NUM_CALLS = 1000;

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    int skipBit = 0x4; // Skip 0b100 when counting upwards.

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 1, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
        REQUIRE((lcv & skipBit) == 0);

        calls++;
    });

    REQUIRE(calls.load() == NUM_CALLS);
}

TEST_CASE("test_qengine_cpu_par_for_skip_wide")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    int NUM_ENTRIES = 2000;
    int NUM_CALLS = 1000;

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    calls.store(0);

    int skipBit = 0x4; // Skip 0b100 when counting upwards.

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 3, [&](const bitCapInt lcv, const int cpu) {
        REQUIRE(lcv < NUM_ENTRIES);
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
        REQUIRE((lcv & skipBit) == 0);

        calls++;
    });
}

TEST_CASE("test_qengine_cpu_par_for_mask")
{
    QEngineCPUPtr qengine = std::make_shared<QEngineCPU>(1, 0);

    int NUM_ENTRIES = 2000;
    int NUM_CALLS = 512; // 2048 >> 2, masked off so all bits are set.

    std::atomic_bool hit[NUM_ENTRIES];
    std::atomic_int calls;

    bitCapInt skipArray[] = { 0x4, 0x100 }; // Skip bits 0b100000100
    int NUM_SKIP = sizeof(skipArray) / sizeof(skipArray[0]);

    calls.store(0);

    for (int i = 0; i < NUM_ENTRIES; i++) {
        hit[i].store(false);
    }

    qengine->SetConcurrencyLevel(1);

    qengine->par_for_mask(0, NUM_ENTRIES, skipArray, 2, [&](const bitCapInt lcv, const int cpu) {
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
        for (int i = 0; i < NUM_SKIP; i++) {
            REQUIRE((lcv & skipArray[i]) == 0);
        }
        calls++;
    });
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cnot")
{
    qftReg->SetPermutation(0x55F00);
    REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    qftReg->CNOT(12, 4, 8);
    REQUIRE_THAT(qftReg, HasProbability(0x55A50));
    qftReg->SetPermutation(0x40001);
    REQUIRE_THAT(qftReg, HasProbability(0x40001));
    qftReg->CNOT(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0xC0001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticnot")
{
    qftReg->SetPermutation(0x55F00);
    REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    qftReg->AntiCNOT(12, 4, 8);
    REQUIRE_THAT(qftReg, HasProbability(0x555A0));
    qftReg->SetPermutation(0x00001);
    REQUIRE_THAT(qftReg, HasProbability(0x00001));
    qftReg->AntiCNOT(18, 19);
    REQUIRE_THAT(qftReg, HasProbability(0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ccnot")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->CCNOT(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCA400));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticcnot")
{
    qftReg->SetPermutation(0xCAC00);
    REQUIRE_THAT(qftReg, HasProbability(0xCAC00));
    qftReg->AntiCCNOT(16, 12, 8, 4);
    REQUIRE_THAT(qftReg, HasProbability(0xCAD00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap")
{
    qftReg->SetPermutation(0xb2000);
    qftReg->Swap(12, 16, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_single_bit")
{
    complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->ApplySingleBit(pauliX, false, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ApplySingleBit(pauliX, false, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_x")
{
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->X(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->X(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_x_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->X(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_y")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->Y(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->Y(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_y_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->Y(1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->Y(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_z")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->Z(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->Z(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_z_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->Z(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cy")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CY(4, 0);
    qftReg->CY(5, 1);
    qftReg->CY(6, 2);
    qftReg->CY(7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));

    qftReg->SetReg(0, 8, 0x10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x10));
    qftReg->H(0);
    qftReg->CY(4, 0);
    qftReg->H(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cy_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CY(4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cz")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CZ(4, 0);
    qftReg->CZ(5, 1);
    qftReg->CZ(6, 2);
    qftReg->CZ(7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cz_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CZ(4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_and")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLAND(0, 0x0c, 4, 4); // 0x0e & 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xce));
    qftReg->SetPermutation(0x3e);
    qftReg->AND(0, 4, 8, 4); // 0xe & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x23e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_or")
{
    qftReg->SetPermutation(0x0c);
    REQUIRE_THAT(qftReg, HasProbability(0x0c));
    qftReg->CLOR(0, 0x0d, 4, 4); // 0x0e | 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xdc));
    qftReg->SetPermutation(0x3e);
    qftReg->OR(0, 4, 8, 4); // 0xe | 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xf3e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xor")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLXOR(0, 0x0d, 4, 4); // 0x0e ^ 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0x3e));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 8, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rt")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->RT(M_PI, 0);
    qftReg->RT(M_PI, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->RT(M_PI, 1);
    qftReg->RT(M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rt_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->RT(M_PI, 1, 2);
    qftReg->RT(M_PI, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rtdyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->RTDyad(1, 1, 0);
    qftReg->RTDyad(1, 1, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->RTDyad(1, 1, 1);
    qftReg->RTDyad(1, 1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rtdyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->RTDyad(1, 1, 1, 2);
    qftReg->RTDyad(1, 1, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crt")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRT(M_PI, 4, 0);
    qftReg->CRT(M_PI, 4, 0);
    qftReg->CRT(M_PI, 5, 1);
    qftReg->CRT(M_PI, 5, 1);
    qftReg->CRT(M_PI, 6, 2);
    qftReg->CRT(M_PI, 6, 2);
    qftReg->CRT(M_PI, 7, 3);
    qftReg->CRT(M_PI, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crt_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRT(M_PI, 4, 0, 4);
    qftReg->CRT(M_PI, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crtdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRTDyad(1, 1, 4, 0);
    qftReg->CRTDyad(1, 1, 4, 0);
    qftReg->CRTDyad(1, 1, 5, 1);
    qftReg->CRTDyad(1, 1, 5, 1);
    qftReg->CRTDyad(1, 1, 6, 2);
    qftReg->CRTDyad(1, 1, 6, 2);
    qftReg->CRTDyad(1, 1, 7, 3);
    qftReg->CRTDyad(1, 1, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crtdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRTDyad(1, 1, 4, 0, 4);
    qftReg->CRTDyad(1, 1, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rx")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RX(M_PI, 0);
    qftReg->RX(M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rx_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RX(M_PI, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rxdyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RXDyad(1, 1, 0);
    qftReg->RXDyad(1, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rxdyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RXDyad(1, 1, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crx")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRX(M_PI, 4, 0);
    qftReg->CRX(M_PI, 5, 1);
    qftReg->CRX(M_PI, 6, 2);
    qftReg->CRX(M_PI, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crx_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRX(M_PI, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crxdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRXDyad(1, 1, 4, 0);
    qftReg->CRXDyad(1, 1, 5, 1);
    qftReg->CRXDyad(1, 1, 6, 2);
    qftReg->CRXDyad(1, 1, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crxdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRXDyad(1, 1, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RY(M_PI, 0);
    qftReg->RY(M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ry_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RY(M_PI, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rydyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RYDyad(1, 1, 0);
    qftReg->RYDyad(1, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rydyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->RYDyad(1, 1, 1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cry")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRY(M_PI, 4, 0);
    qftReg->CRY(M_PI, 5, 1);
    qftReg->CRY(M_PI, 6, 2);
    qftReg->CRY(M_PI, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cry_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRY(M_PI, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crydyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRYDyad(1, 1, 4, 0);
    qftReg->CRYDyad(1, 1, 5, 1);
    qftReg->CRYDyad(1, 1, 6, 2);
    qftReg->CRYDyad(1, 1, 7, 3);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crydyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->CRYDyad(1, 1, 4, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rz")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->RZ(M_PI, 1);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rz_reg")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZ(M_PI, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rzdyad")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZDyad(1, 1, 1);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rzdyad_reg")
{
    qftReg->SetReg(0, 8, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(0, 2);
    qftReg->RZDyad(1, 1, 0, 2);
    qftReg->H(0, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crz")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZ(M_PI, 4, 0);
    qftReg->CRZ(M_PI, 5, 1);
    qftReg->CRZ(M_PI, 6, 2);
    qftReg->CRZ(M_PI, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crz_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZ(M_PI, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crzdyad")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZDyad(1, 1, 4, 0);
    qftReg->CRZDyad(1, 1, 5, 1);
    qftReg->CRZDyad(1, 1, 6, 2);
    qftReg->CRZDyad(1, 1, 7, 3);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_crzdyad_reg")
{
    qftReg->SetReg(0, 8, 0x35);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x35));
    qftReg->H(0, 4);
    qftReg->CRZDyad(1, 1, 4, 0, 4);
    qftReg->H(0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x36));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_exp")
{
    qftReg->SetPermutation(0x80001);
    qftReg->Exp(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_exp_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->Exp(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expdyad")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpDyad(4, 1, 19);
    qftReg->SetPermutation(0x80001);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expdyad_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expx")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpX(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ExpX(2.0 * M_PI, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expx_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpX(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expxdyad")
{
    qftReg->SetPermutation(0x80001);
    qftReg->ExpXDyad(4, 1, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 1));
    qftReg->ExpXDyad(4, 1, 19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expxdyad_reg")
{
    qftReg->SetPermutation(0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpXDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expy")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->ExpY(2.0 * M_PI, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->ExpY(2.0 * M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expy_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpY(2.0 * M_PI, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpY(2.0 * M_PI, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expydyad")
{
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
    qftReg->ExpYDyad(4, 1, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->H(1);
    qftReg->ExpYDyad(4, 1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expydyad_reg")
{
    qftReg->SetReg(0, 8, 0x13);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));
    qftReg->ExpYDyad(4, 1, 1, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x0d));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpYDyad(4, 1, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expz")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->ExpZ(2.0 * M_PI, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->ExpZ(2.0 * M_PI, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expz_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpZ(2.0 * M_PI, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expzdyad")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->ExpZDyad(4, 1, 0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->ExpZDyad(4, 1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expzdyad_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->ExpZDyad(4, 1, 1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rol")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ROL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(3));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ror")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ROR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(192));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_asl")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ASL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(66));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_asr")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->ASR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(96));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsl")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->LSL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_lsr")
{
    qftReg->SetPermutation(129);
    REQUIRE_THAT(qftReg, HasProbability(129));
    qftReg->LSR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(64));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_inc")
{
    int i;

    qftReg->SetPermutation(250);
    for (i = 0; i < 8; i++) {
        qftReg->INC(1, 0, 8);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incs")
{
    int i;

    qftReg->SetPermutation(250);
    for (i = 0; i < 8; i++) {
        qftReg->INCS(1, 0, 8, 9);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incc")
{
    int i;

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        qftReg->INCC(1, 0, 8, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 2 + i - 8));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcd")
{
    int i;

    qftReg->SetPermutation(0x95);
    for (i = 0; i < 8; i++) {
        qftReg->INCBCD(1, 0, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x96 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 4));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incbcdc")
{
    int i;

    qftReg->SetPermutation(0x095);
    for (i = 0; i < 8; i++) {
        qftReg->INCBCDC(1, 0, 8, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x096 + i));
        } else if (i == 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02 + i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incsc")
{
    int i;

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        qftReg->INCSC(1, 0, 8, 9, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 2 + i - 8));
        }
    }

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        qftReg->INCSC(1, 0, 8, 8);
        if (i < 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 10, 2 + i - 8));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dec")
{
    int i;
    int start = 0x08;

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        qftReg->DEC(9, 0, 8);
        start -= 9;
        REQUIRE_THAT(qftReg, HasProbability(0, 19, 0xff - i * 9));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decs")
{
    int i;
    int start = 0x08;

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        qftReg->DECS(9, 0, 8, 9);
        start -= 9;
        REQUIRE_THAT(qftReg, HasProbability(0, 19, 0xff - i * 9));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decc")
{
    int i;

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        qftReg->DECC(1, 0, 8, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcd")
{
    int i;

    qftReg->SetPermutation(0x94);
    for (i = 0; i < 8; i++) {
        qftReg->DECBCD(1, 0, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x93 - i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x89 - i + 4));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decbcdc")
{
    int i;

    qftReg->SetPermutation(0x005);
    for (i = 0; i < 8; i++) {
        qftReg->DECBCDC(1, 0, 8, 8);
        if (i < 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x103 - i));
        } else if (i == 4) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x099));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x197 - i + 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decsc")
{
    int i;

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        qftReg->DECSC(1, 0, 8, 9, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }

    qftReg->SetPermutation(7);
    for (i = 0; i < 10; i++) {
        qftReg->DECSC(1, 0, 8, 8);
        if (i < 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 5 - i + 256));
        } else if (i == 6) {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 0xff));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 9, 253 - i + 7 + 256));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mul")
{
    int i;

    qftReg->SetPermutation(3);
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        qftReg->MUL(2, 0, 8, 8, true);
        res *= 2;
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        res &= 255;
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_div")
{
    int i;

    qftReg->SetPermutation(256);
    bitCapInt res = 256;
    for (i = 0; i < 8; i++) {
        qftReg->DIV(2, 0, 8, 8);
        res /= 2;
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cmul")
{
    int i;

    qftReg->SetPermutation(3 | (1 << 16));
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        qftReg->CMUL(2, 0, 8, 16, 8, true);
        if ((i % 2) == 0) {
            res *= 2;
        }
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        res &= 255;
        qftReg->X(16);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cdiv")
{
    int i;

    qftReg->SetPermutation(256 | (1 << 16));
    bitCapInt res = 256;
    for (i = 0; i < 8; i++) {
        qftReg->CDIV(2, 0, 8, 16, 8);
        if ((i % 2) == 0) {
            res /= 2;
        }
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        qftReg->X(16);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qft_h")
{
    real1 qftProbs[20];
    qftReg->SetPermutation(85);

    int i, j;

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    qftReg->QFT(0, 8);

    qftReg->IQFT(0, 8);

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, 8, 85));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_zero_phase_flip")
{
    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1);
    qftReg->ZeroPhaseFlip(1, 1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_c_phase_flip_if_less")
{
    qftReg->SetReg(0, 20, 0x40000);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x40000));
    qftReg->H(19);
    qftReg->CPhaseFlipIfLess(1, 19, 1, 18);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xC0000));

    qftReg->SetReg(0, 20, 0x00);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00000));
    qftReg->H(19);
    qftReg->CPhaseFlipIfLess(1, 19, 1, 18);
    qftReg->H(19);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_phase_flip")
{
    qftReg->SetReg(0, 8, 0x00);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    qftReg->PhaseFlip();
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m")
{
    REQUIRE(qftReg->M(0) == 0);
    qftReg->SetReg(0, 8, 0x03);
    REQUIRE(qftReg->M(0) == true);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mreg")
{
    qftReg->SetReg(0, 8, 0);
    REQUIRE(qftReg->MReg(0, 8) == 0);
    qftReg->SetReg(0, 8, 0x2b);
    REQUIRE(qftReg->MReg(0, 8) == 0x2b);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg")
{
    int j;

    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    qftReg->IndexedLDA(0, 8, 8, 8, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x303));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0));

    qftReg->H(8, 8);
    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }

    qftReg->IndexedLDA(8, 8, 0, 8, testPage);

    for (j = 0; j < 256; j++) {
        testPage[j] = 255 - j;
    }
    qftReg->IndexedADC(8, 8, 0, 8, 16, testPage);
    qftReg->H(8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 0xff));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(1 << 16);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1 << 16));

    qftReg->H(8, 8);
    unsigned char* testPage = cl_alloc(256);
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    qftReg->IndexedLDA(8, 8, 0, 8, testPage);

    qftReg->IndexedSBC(8, 8, 0, 8, 16, testPage);
    qftReg->H(8, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 1 << 16));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg_long")
{
    int j;

    qftReg->SetReg(0, 9, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    qftReg->IndexedLDA(0, 9, 9, 9, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 17, 0x603));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg_long_index")
{
    int j;

    qftReg->SetPermutation(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 18, 0));

    qftReg->H(9, 9);
    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }

    qftReg->IndexedLDA(9, 9, 0, 9, testPage);

    for (j = 0; j < 512; j++) {
        testPage[j * 2] = (511 - j) & 0xff;
        testPage[j * 2 + 1] = ((511 - j) & 0x100) >> 8;
    }
    qftReg->IndexedADC(9, 9, 0, 9, 18, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 9, 0x1ff));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg_long_index")
{
    int j;

    qftReg->SetPermutation(1 << 18);
    REQUIRE_THAT(qftReg, HasProbability(0, 18, 1 << 18));

    qftReg->H(9, 9);
    unsigned char* testPage = cl_alloc(1024);
    for (j = 0; j < 512; j++) {
        testPage[j * 2] = j & 0xff;
        testPage[j * 2 + 1] = (j & 0x100) >> 8;
    }
    qftReg->IndexedLDA(9, 9, 0, 9, testPage);

    qftReg->IndexedSBC(9, 9, 0, 9, 18, testPage);
    qftReg->H(9, 9);
    REQUIRE_THAT(qftReg, HasProbability(0, 19, 1 << 18));
    free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decohere")
{
    QInterfacePtr qftReg2 = CreateQuantumInterface(testSubEngineType, testSubEngineType, 4, 0, rng);

    qftReg->SetPermutation(0x2b);
    qftReg->Decohere(0, 4, qftReg2);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dispose")
{
    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(0, 4);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));

    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(4, 4);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cohere")
{
    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, 4, 0x0b, rng);
    QInterfacePtr qftReg2 = CreateQuantumInterface(testSubEngineType, testSubEngineType, 4, 0x02, rng);
    qftReg->Cohere(qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_setbit")
{
    qftReg->SetPermutation(0x02);
    qftReg->SetBit(0, true);
    qftReg->SetBit(1, false);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_proball")
{
    qftReg->SetPermutation(0x02);
    REQUIRE(qftReg->ProbAll(0x02) > 0.99);
    REQUIRE(qftReg->ProbAll(0x03) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_grover")
{
    int i;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.

    const int TARGET_PROB = 100;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(0, 8);

    // std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        qftReg->DEC(100, 0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->INC(100, 0, 8);
        // This ends the "oracle."
        qftReg->H(0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->H(0, 8);
        qftReg->PhaseFlip();
        // std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qftReg->ProbAll(TARGET_PROB) << std::endl;
    }

    // std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    // std::cout << "Full Result:    " << qftReg << std::endl;
    // std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_grover_lookup")
{
    int i;

    // Grover's search to find a value in a lookup table.
    // We search for 100. All values in lookup table are 1 except a single match.

    const int TARGET_PROB = 100 + (230 << 8);

    unsigned char* toLoad = cl_alloc(256);
    for (i = 0; i < 256; i++) {
        toLoad[i] = 1;
    }
    toLoad[230] = 100;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(8, 8);
    qftReg->IndexedLDA(8, 8, 0, 8, toLoad);

    // std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        qftReg->DEC(100, 0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->INC(100, 0, 8);
        // This ends the "oracle."
        qftReg->X(16);
        qftReg->IndexedSBC(8, 8, 0, 8, 16, toLoad);
        qftReg->X(16);
        qftReg->H(8, 8);
        qftReg->ZeroPhaseFlip(8, 8);
        qftReg->H(8, 8);
        qftReg->PhaseFlip();
        qftReg->IndexedADC(8, 8, 0, 8, 16, toLoad);
        // std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qftReg->ProbAll(TARGET_PROB) << std::endl;
    }

    // std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    // std::cout << "Full Result:    " << qftReg << std::endl;
    // std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
    free(toLoad);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_minimization")
{
    const bitLenInt length = 8;
    bitCapInt threshold = 1 << (length - 2);
    int i;

    qftReg->SetPermutation(0);
    qftReg->H(0, length);

    // Assume the function to minimize is (close to) one-to-one. For 4 possible inputs, with 1 desired output, one
    // Grover's search iteration will return the exact desired output. Selecting the bottom quarter of outputs as the
    // (degenerate) desired result, one iteration of Grover's search can return them in equal superposition exactly,
    // using "PhaseFlipIfLess()." We repeat a degenerate version of Grover's search with four equiprobable inputs, one
    // desired output, found with one Grover's iteration, until all degeneracy is removed, and then we have the single
    // optimal state.

    // Move from inputs to outputs. (We're looking for the minimum of NOT on all bits.)
    qftReg->X(0, length);

    for (i = 0; i < (length / 2); i++) {
        // Phase flip the desired outputs:
        qftReg->PhaseFlipIfLess(threshold, 0, length);

        // Next, we want to phase flip the original input.
        // If the function is one-to-one, H() takes the superposition of outputs back to zero, (except for the phase
        // effects we're leveraging). At each iteration, we expect an equal superposition of the 1/(1<<(2 *i)) low
        // fraction of outputs and 0 probability in all other states. This is the one and only pure state that we want
        // to flip the phase of, at each iteration.
        qftReg->H(0, length - (i * 2));
        qftReg->ZeroPhaseFlip(0, length);
        qftReg->H(0, length - (i * 2));

        // Phase flip the entire state, to conclude the Grover's iteration:
        // qftReg->PhaseFlip();

        // Now we have one quarter as many states to look for, in the ideal, and we've returned the state to totally
        // in-phase.
        threshold >>= 2;
    }

    // Move back from outputs to inputs. (We're looking for the minimum of NOT on all bits.)
    qftReg->X(0, length);

    // The state should usually be close to the global minimum output, now.
    // If the function is one-to-one, it should return the exact minimum.
    REQUIRE_THAT(qftReg, HasProbability(0, 8, (1 << length) - 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fast_grover")
{
    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    const bitLenInt length = 10;
    const int TARGET_PROB = 100;
    int i;
    bitLenInt partLength;

    // Start in a superposition of all inputs.
    qftReg->SetPermutation(0);
    qftReg->H(0, length);

    // Our black box "oracle" secretly returns true for TARGET_PROB and false for all other inputs. This is the function
    // we are trying to invert.
    for (i = 0; i < (length / 2); i++) {
        // This is the number of bits not yet fixed.
        partLength = length - (i * 2);

        // We map from input to output.
        qftReg->DEC(TARGET_PROB, 0, length);
        // Phase flip the target state.
        qftReg->ZeroPhaseFlip(0, partLength);
        // We map back from outputs to inputs.
        qftReg->INC(TARGET_PROB, 0, length);

        // Phase flip the input state from the previous iteration.
        qftReg->H(0, partLength);
        qftReg->ZeroPhaseFlip(0, partLength);
        qftReg->H(0, partLength);

        // Now, we have one quarter as many states to look for.
    }

    REQUIRE_THAT(qftReg, HasProbability(0, length, TARGET_PROB));
}

void ExpMod(QInterfacePtr qftReg, bitCapInt base, bitLenInt baseStart, bitLenInt baseLen, bitLenInt expStart,
    bitLenInt expLen, bitLenInt carryStart, bitLenInt recordStart)
{
    bitCapInt workingPower = base;
    bitLenInt regStart1, regStart2;
    bitLenInt i;
    for (i = 0; i < expLen; i++) {
        if (i & 1) {
            regStart1 = carryStart;
            regStart2 = baseStart;
        } else {
            regStart1 = baseStart;
            regStart2 = carryStart;
        }
        qftReg->CMUL(workingPower, regStart1, recordStart, expStart + i, baseLen, false);
        qftReg->CNOT(regStart1, regStart2, baseLen);
        qftReg->CDIV(workingPower, regStart1, recordStart, expStart + i, baseLen);
        qftReg->SetReg(regStart1, baseLen, 0);
        workingPower *= base;
    }
    if (i & 1) {
        qftReg->CNOT(carryStart, baseStart, baseLen);
        qftReg->SetReg(carryStart, baseLen, 0);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_expmod")
{
    bitLenInt baseLen = 4;
    bitLenInt expLen = 2;
    qftReg->SetPermutation(1);
    // Last bits are exponent:
    qftReg->SetReg(20 - expLen, expLen, 2);
    ExpMod(qftReg, 5, 0, baseLen, 20 - expLen, expLen, 2 * baseLen, baseLen);
    REQUIRE_THAT(qftReg, HasProbability(0, baseLen, 25));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_set_reg")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
    qftReg->SetReg(0, 8, 10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 10));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_basis_change")
{
    int i;
    unsigned char* toSearch = cl_alloc(256);

    // Create the lookup table
    for (i = 0; i < 256; i++) {
        toSearch[i] = 100;
    }

    // Divide qftReg into two registers of 8 bits each
    qftReg->SetPermutation(0);
    qftReg->H(8, 8);
    qftReg->IndexedLDA(8, 8, 0, 8, toSearch);
    qftReg->H(8, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, 100));
    free(toSearch);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_entanglement")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x0));
    for (int i = 0; i < qftReg->GetQubitCount(); i += 2) {
        qftReg->X(i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x55555));
    for (int i = 0; i < (qftReg->GetQubitCount() - 1); i += 2) {
        qftReg->CNOT(i, i + 1);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xfffff));
    for (int i = qftReg->GetQubitCount() - 2; i > 0; i -= 2) {
        qftReg->CNOT(i - 1, i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xAAAAB));
    for (int i = 1; i < qftReg->GetQubitCount(); i += 2) {
        qftReg->X(i);
    }
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_bit")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->Swap(0, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap_reg")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->Swap(0, 1, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}
