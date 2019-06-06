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
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "qfactory.hpp"
#include "qneuron.hpp"

#include "tests.hpp"

using namespace Qrack;

#define ALIGN_SIZE 64
#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1 __tmp_a = A;                                                                                             \
        real1 __tmp_b = B;                                                                                             \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_a > (__tmp_b - EPSILON));                                                                        \
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
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
    return (unsigned char*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (unsigned char*)_aligned_malloc(
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount),
        ALIGN_SIZE);
#else
    return (unsigned char*)aligned_alloc(ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
#endif
}

void cl_free(void* toFree)
{
    if (toFree) {
#if defined(_WIN32)
        _aligned_free(toFree);
#else
        free(toFree);
#endif
    }
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

    cmplx3 = std::polar(1.0, M_PI / 2.0);
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

    const int NUM_ENTRIES = 2000;
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

    const int NUM_ENTRIES = 2000;
    const int NUM_CALLS = 1000;

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

    const int NUM_ENTRIES = 2000;
    const int NUM_CALLS = 1000;

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

    const int NUM_ENTRIES = 2000;
    const int NUM_CALLS = 512; // 2048 >> 2, masked off so all bits are set.

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

#if ENABLE_OPENCL
TEST_CASE_METHOD(QInterfaceTestFixture, "test_oclengine")
{
    if (testEngineType == QINTERFACE_OPENCL) {
        std::vector<DeviceContextPtr> devices = OCLEngine::Instance()->GetDeviceContextPtrVector();
        REQUIRE(devices.size() > 0);

        OCLEngine::Instance()->SetDefaultDeviceContext(OCLEngine::Instance()->GetDeviceContextPtr(-1));

        CHECK_THROWS(OCLEngine::Instance()->GetDeviceContextPtr(-2));

        Qrack::OCLEngine::InitOCL(true, true, "_test_ocl_kernel_compile/");
        
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_change_device")
{
    if (testEngineType == QINTERFACE_OPENCL) {
        qftReg->SetPermutation(0x55F00);
        REQUIRE_THAT(qftReg, HasProbability(0x55F00));
        std::dynamic_pointer_cast<QEngineOCL>(qftReg)->SetDevice(0);
        REQUIRE_THAT(qftReg, HasProbability(0x55F00));
    }
}
#endif

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qengine_getmaxqpower")
{
    // Assuming default engine has 20 qubits:
    REQUIRE((qftReg->GetMaxQPower() == 1048576U));
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtswap")
{
    qftReg->SetPermutation(0xb2000);
    qftReg->SqrtSwap(12, 16, 4);
    qftReg->SqrtSwap(12, 16, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_isqrtswap")
{
    qftReg->SetPermutation(0xb2000);
    qftReg->SqrtSwap(12, 16, 4);
    qftReg->ISqrtSwap(12, 16, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xb2000));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->CSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->SetPermutation(0x101);
    qftReg->CSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
    qftReg->H(8);
    qftReg->CSwap(control, 1, 0, 4);
    qftReg->CSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->AntiCSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
    qftReg->H(8);
    qftReg->AntiCSwap(control, 1, 0, 4);
    qftReg->AntiCSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_csqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->SetPermutation(0x101);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
    qftReg->H(8);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x110));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticsqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
    qftReg->H(8);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x010));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cisqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x101);
    qftReg->CSqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
    qftReg->H(8);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->CISqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x101));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_anticisqrtswap")
{
    bitLenInt control[1] = { 8 };
    qftReg->SetPermutation(0x001);
    qftReg->AntiCSqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
    qftReg->H(8);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->AntiCISqrtSwap(control, 1, 0, 4);
    qftReg->H(8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x001));
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_controlled_single_bit")
{
    complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x8000F);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x8000F));
    qftReg->ApplyControlledSingleBit(controls, 3, 19, pauliX);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x0F));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->ApplyControlledSingleBit(controls, 3, 19, pauliX);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->ApplyControlledSingleBit(controls, 3, 19, pauliX);
    qftReg->ApplyControlledSingleBit(controls, 3, 19, pauliX);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_apply_anticontrolled_single_bit")
{
    complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    bitLenInt controls[3] = { 0, 1, 3 };
    qftReg->SetPermutation(0x80000);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80000));
    qftReg->ApplyAntiControlledSingleBit(controls, 3, 19, pauliX);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x00));
    qftReg->SetPermutation(0x80001);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->ApplyAntiControlledSingleBit(controls, 3, 19, pauliX);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    qftReg->ApplyAntiControlledSingleBit(controls, 3, 19, pauliX);
    qftReg->ApplyAntiControlledSingleBit(controls, 3, 19, pauliX);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->H(3);
    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x80001));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_s")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->S(0);
    qftReg->S(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->S(1);
    qftReg->S(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_s_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->S(1, 2);
    qftReg->S(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_is")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->S(0);
    qftReg->IS(0);
    qftReg->IS(1);
    qftReg->S(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_is_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->S(1, 2);
    qftReg->IS(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_t")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->T(0);
    qftReg->H(0);
    qftReg->H(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->T(1);
    qftReg->H(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_t_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->T(1, 2);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x04));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_it")
{
    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->T(0);
    qftReg->IT(0);
    qftReg->IT(1);
    qftReg->T(1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_it_reg")
{
    qftReg->SetReg(0, 8, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->T(1, 2);
    qftReg->IT(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
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
    qftReg->X(4, 1);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x1d));
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
    qftReg->SetPermutation(0x03);
    qftReg->AND(0, 0, 8, 4); // 0x3 & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x303));
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
    qftReg->SetPermutation(0x03);
    qftReg->AND(0, 0, 8, 4); // 0x3 & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x303));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xor")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLXOR(0, 0x0d, 4, 4); // 0x0e ^ 0x0d
    REQUIRE_THAT(qftReg, HasProbability(0x3e));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 8, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
    qftReg->SetPermutation(0xe);
    qftReg->XOR(0, 0, 0, 4); // 0xe ^ 0xe
    REQUIRE_THAT(qftReg, HasProbability(0x0));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 0, 4); // 0xe ^ 0xe
    REQUIRE_THAT(qftReg, HasProbability(0x3d));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 4, 4); // 0xe ^ 0xe
    REQUIRE_THAT(qftReg, HasProbability(0xde));
    qftReg->SetPermutation(0x0e);
    qftReg->CLXOR(0, 0x0d, 0, 4); // 0x0e ^ 0x0d
    REQUIRE_THAT(qftReg, HasProbability(0x03));
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniform_cry")
{
    bitLenInt controls[2] = { 4, 5 };
    real1 angles[4] = { M_PI, M_PI, 0, 0 };

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledRY(NULL, 0, 0, angles);
    qftReg->UniformlyControlledRY(NULL, 0, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    qftReg->SetReg(0, 8, 0x12);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x12));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));

    controls[0] = 5;
    controls[1] = 4;

    qftReg->SetReg(0, 8, 0x22);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x22));
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));

    controls[0] = 4;
    controls[1] = 5;

    qftReg->SetReg(0, 8, 0x02);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    qftReg->H(4);
    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg->UniformlyControlledRY(controls, 2, 1, angles);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));

    complex pauliRY[4];
    real1 cosine, sine;
    bitLenInt i, j;

    qftReg->SetReg(0, 8, 0x02);
    QInterfacePtr qftReg2 = qftReg->Clone();
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x02));
    REQUIRE_THAT(qftReg2, HasProbability(0, 8, 0x02));

    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg2->QInterface::UniformlyControlledRY(controls, 2, 0, angles);

    REQUIRE(qftReg->ApproxCompare(qftReg2));
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_uniform_crz")
{
    bitLenInt controls[2] = { 4, 5 };
    real1 angles[4] = { M_PI, M_PI, 0, 0 };

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(NULL, 0, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));

    qftReg->SetReg(0, 8, 0x11);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x11));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x13));

    qftReg->SetReg(0, 8, 0x21);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));

    controls[0] = 5;
    controls[1] = 4;

    qftReg->SetReg(0, 8, 0x21);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x21));
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x23));

    controls[0] = 4;
    controls[1] = 5;

    qftReg->SetReg(0, 8, 0x01);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x01));
    qftReg->H(4);
    qftReg->H(1, 2);
    qftReg->UniformlyControlledRZ(controls, 2, 1, angles);
    qftReg->H(1, 2);
    qftReg->H(4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x03));
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cinc")
{
    int i;

    qftReg->SetPermutation(1);
    qftReg->CINC(1, 0, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    bitLenInt controls[1] = { 8 };

    qftReg->SetPermutation(250);

    for (i = 0; i < 8; i++) {
        // Turn control on
        qftReg->X(controls[0]);

        qftReg->CINC(1, 0, 8, controls, 1);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }

        // Turn control off
        qftReg->X(controls[0]);

        qftReg->CINC(1, 0, 8, controls, 1);
        if (i < 5) {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(0, 8, i - 5));
        }
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dec")
{
    int i;
    int start = 0x08;

    qftReg->SetPermutation(2);
    qftReg->CDEC(1, 0, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));

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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cdec")
{
    int i;

    bitLenInt controls[1] = { 8 };

    qftReg->SetPermutation(0x08);
    for (i = 0; i < 8; i++) {
        // Turn control on
        qftReg->X(controls[0]);

        qftReg->CDEC(9, 0, 8, controls, 1);
        REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xff - i * 9));

        // Turn control off
        qftReg->X(controls[0]);

        qftReg->CDEC(9, 0, 8, controls, 1);
        REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xff - i * 9));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_mul")
{
    int i;

    qftReg->SetPermutation(3);
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        qftReg->SetReg(8, 8, 0x00);
        qftReg->MUL(2, 0, 8, 8);
        res &= 0xFF;
        res *= 2;
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
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

    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(1);
    qftReg->CMUL(2, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 2));

    qftReg->SetPermutation(3 | (1 << 16));
    bitCapInt res = 3;
    for (i = 0; i < 8; i++) {
        qftReg->CMUL(2, 0, 8, 8, controls, 1);
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

    bitLenInt controls[1] = { 16 };

    qftReg->SetPermutation(2);
    qftReg->CDIV(2, 0, 8, 8, NULL, 0);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1));

    qftReg->SetPermutation(256 | (1 << 16));
    bitCapInt res = 256;
    for (i = 0; i < 8; i++) {
        qftReg->CDIV(2, 0, 8, 8, controls, 1);
        if ((i % 2) == 0) {
            res /= 2;
        }
        REQUIRE_THAT(qftReg, HasProbability(0, 16, res));
        qftReg->X(16);
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qft_h")
{
    qftReg->SetPermutation(85);

    int i;

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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_tryseparate")
{
    qftReg->SetPermutation(85);

    int i;

    qftReg->QFT(0, 8);

    qftReg->IQFT(0, 8);

    for (i = 0; i < 8; i++) {
        qftReg->TrySeparate(i);
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m_array")
{
    bitLenInt bits[3] = { 0, 2, 3 };
    REQUIRE(qftReg->M(0) == 0);
    qftReg->SetReg(0, 8, 0x07);
    REQUIRE(qftReg->M(bits, 3) == 5);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x07));
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
    cl_free(testPage);
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
    cl_free(testPage);
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
    cl_free(testPage);
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
    cl_free(testPage);
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
    cl_free(testPage);
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
    cl_free(testPage);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_clone")
{
    qftReg->SetPermutation(0x2b);
    QInterfacePtr qftReg2 = qftReg->Clone();
    qftReg2->X(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0x2b));
    REQUIRE_THAT(qftReg2, HasProbability(0, 20, 0xd4));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decompose")
{
    QInterfacePtr qftReg2 = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0, rng);

    qftReg->SetPermutation(0x2b);
    qftReg->Decompose(0, 4, qftReg2);

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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_compose")
{
    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0x0b, rng);
    QInterfacePtr qftReg2 =
        CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0x02, rng);
    qftReg->Compose(qftReg2);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_trydecompose")
{
    QInterfacePtr qftReg2 = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0, rng);

    qftReg->SetPermutation(0x2b);
    REQUIRE(qftReg->TryDecompose(0, 4, qftReg2) == true);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));

    qftReg->SetPermutation(0xb);
    qftReg->H(0, 4);
    qftReg->CNOT(0, 4, 4);
    REQUIRE(qftReg->TryDecompose(0, 4, qftReg2) == false);
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probreg")
{
    qftReg->SetPermutation(0x20);
    REQUIRE(qftReg->ProbReg(4, 4, 0x2) > 0.99);
    REQUIRE(qftReg->ProbReg(4, 4, 0x3) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_probmask")
{
    qftReg->SetPermutation(0x21);
    REQUIRE(qftReg->ProbMask(0xF0, 0x20) > 0.99);
    REQUIRE(qftReg->ProbMask(0xF0, 0x40) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_forcem")
{
    qftReg->SetPermutation(0x0);
    qftReg->H(0, 4);

    REQUIRE_FLOAT(qftReg->ProbMask(0xF, 0), 0.0625);
    REQUIRE_FLOAT(qftReg->ProbMask(0x7, 0), 0.125);

    bitLenInt bits[3] = { 0, 1, 2 };
    bool results[3] = { 0, 1, 0 };

    qftReg->ForceM(bits, 3, results);

    REQUIRE(qftReg->ProbMask(0x7, 0x2) > 0.99);
    REQUIRE_FLOAT(qftReg->ProbMask(0xF, 0x2), 0.5);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_getamplitude")
{
    qftReg->SetPermutation(0x03);
    qftReg->H(0, 2);
    REQUIRE(norm((qftReg->GetAmplitude(0x01)) + (qftReg->GetAmplitude(0x03))) < 0.01);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_getquantumstate")
{
    complex state[1U << 4U];
    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 4, 0x0b, rng);
    qftReg->GetQuantumState(state);
    qftReg->SetQuantumState(state);
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

    const bitLenInt indexLength = 8;
    const bitLenInt valueLength = 8;
    const bitLenInt carryIndex = indexLength + valueLength;
    const int TARGET_VALUE = 100;
    const int TARGET_KEY = 230;

    unsigned char* toLoad = cl_alloc(1 << indexLength);
    for (i = 0; i < (1 << indexLength); i++) {
        toLoad[i] = 1;
    }
    toLoad[TARGET_KEY] = TARGET_VALUE;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(valueLength, indexLength);
    qftReg->IndexedLDA(valueLength, indexLength, 0, valueLength, toLoad);

    // Twelve iterations maximizes the probablity for 256 searched elements, for example.
    // For an arbitrary number of qubits, this gives the number of iterations for optimal probability.
    int optIter = M_PI / (4.0 * asin(1.0 / sqrt(1 << indexLength)));

    for (i = 0; i < optIter; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        qftReg->DEC(TARGET_VALUE, 0, valueLength);
        qftReg->ZeroPhaseFlip(0, valueLength);
        qftReg->INC(TARGET_VALUE, 0, valueLength);
        // This ends the "oracle."
        qftReg->X(carryIndex);
        qftReg->IndexedSBC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        qftReg->X(carryIndex);
        qftReg->H(valueLength, indexLength);
        qftReg->ZeroPhaseFlip(valueLength, indexLength);
        qftReg->H(valueLength, indexLength);
        // qftReg->PhaseFlip();
        qftReg->IndexedADC(valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
    }

    REQUIRE_THAT(qftReg, HasProbability(0, indexLength + valueLength, TARGET_VALUE | (TARGET_KEY << valueLength)));
    cl_free(toLoad);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_fast_grover")
{
    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.
    const bitLenInt length = 10;
    const int TARGET_PROB = 100;
    int i;
    bitLenInt partStart;
    // Start in a superposition of all inputs.
    qftReg->SetPermutation(0);
    // For Grover's search, our black box "oracle" would secretly return true for TARGET_PROB and false for all other
    // inputs. This is the function we are trying to invert. For an improvement in search speed, we require n/2 oracles
    // for an n bit search target. Each oracle marks 2 bits of the n total. This method might be applied to an ORDERED
    // lookup table search, in which a series of quaternary decisions can ultimately select any result in the list.
    for (i = 0; i < (length / 2); i++) {
        // This is the number of bits not yet fixed.
        partStart = length - ((i + 1) * 2);
        qftReg->H(partStart, 2);
        // We map from input to output.
        qftReg->DEC(TARGET_PROB & (3 << partStart), 0, length);
        // Phase flip the target state.
        qftReg->ZeroPhaseFlip(partStart, 2);
        // We map back from outputs to inputs.
        qftReg->INC(TARGET_PROB & (3 << partStart), 0, length);
        // Phase flip the input state from the previous iteration.
        qftReg->H(partStart, 2);
        qftReg->ZeroPhaseFlip(partStart, 2);
        qftReg->H(partStart, 2);
        // Now, we have one quarter as many states to look for.
    }

    REQUIRE_THAT(qftReg, HasProbability(0, length, TARGET_PROB));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_quaternary_search")
{
    bitLenInt i;
    bitLenInt partStart;
    bitLenInt partLength;

    // Grover's search to find a value in an ordered list.

    const bitLenInt indexLength = 6;
    const bitLenInt valueLength = 6;
    const bitLenInt carryIndex = 19;
    const int TARGET_VALUE = 6;
    const int TARGET_KEY = 18;

    bool foundPerm = false;

    unsigned char* toLoad = cl_alloc(1 << indexLength);
    for (i = 0; i < TARGET_KEY; i++) {
        toLoad[i] = 2;
    }
    toLoad[TARGET_KEY] = TARGET_VALUE;
    for (i = (TARGET_KEY + 1); i < (1 << indexLength); i++) {
        toLoad[i] = 7;
    }

    qftReg->SetPermutation(0);
    partLength = indexLength;

    for (i = 0; i < (indexLength / 2); i++) {
        // We're in an exact permutation basis state, at this point, unless more than one quadrant contained a match for
        // our search target on the previous iteration. We can check the quadrant boundaries, without disturbing the
        // state. If there was more than one match, we either collapse into a valid state, so that we can continue as
        // expected for one matching quadrant, or we collapse into an identifiably invalid set of bounds that cannot
        // contain our match, which can be identified by checking two values and proceeding with special case logic.

        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt unfixedMask = (1 << unfixedLength) - 1;
        bitCapInt key = (qftReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);

        // (We could either manipulate the quantum bits directly to check this, or rely on auxiliary classical computing
        // components, as need and efficiency dictate).
        bitCapInt lowBound = toLoad[key];
        bitCapInt highBound = toLoad[key | unfixedMask];

        if (lowBound == TARGET_VALUE) {
            // We've found our match, and the key register already contains the correct value.
            std::cout << "Is low bound";
            foundPerm = true;
            break;
        } else if (highBound == TARGET_VALUE) {
            // We've found our match, but our key register points to the opposite bound.
            std::cout << "Is high bound";
            qftReg->X(2 * valueLength, partLength);
            foundPerm = true;
            break;
        } else if (((lowBound < TARGET_VALUE) && (highBound < TARGET_VALUE)) ||
            ((lowBound > TARGET_VALUE) && (highBound > TARGET_VALUE))) {
            // If we measure the key as a quadrant that doesn't contain our value, then either there is more than one
            // quadrant with bounds that match our target value, or there is no match to our target in the list.
            foundPerm = false;
            break;
        }

        // Prepare partial index superposition, of two most significant qubits that have not yet been fixed:
        partLength = indexLength - ((i + 1) * 2);
        partStart = (2 * valueLength) + partLength;
        qftReg->H(partStart, 2);

        // Load lower bound of quadrants:
        qftReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength, carryIndex, toLoad);

        if (partLength > 0) {
            // In this branch, our quadrant is "degenerate," (we mean, having more than one key/value pair).

            // Load upper bound of quadrants:
            qftReg->X(2 * valueLength, partLength);
            qftReg->IndexedADC(2 * valueLength, indexLength, valueLength, valueLength, carryIndex, toLoad);

            // Our "oracle" is true if the target is in this quadrant, and false otherwise:
            // Flip phase if lower bound <= the target value.
            qftReg->PhaseFlipIfLess(TARGET_VALUE + 1, 0, valueLength);
            // Flip phase if upper bound < the target value.
            qftReg->PhaseFlipIfLess(TARGET_VALUE, valueLength, valueLength);
            // If both are higher, this is not the quadrant, and neither flips the permutation phase.
            // If both are lower, this is not the quadrant, and the 2 phase flips of the permutation cancel.
            // If both match the target, the above still tags the quadrant.
        } else {
            // In this branch, we have one key/value pair in each quadrant, so we can use our usual Grover's oracle.

            // We map from input to output.
            qftReg->DEC(TARGET_VALUE, 0, valueLength);
            // Phase flip the target state.
            qftReg->ZeroPhaseFlip(0, valueLength);
            // We map back from outputs to inputs.
            qftReg->INC(TARGET_VALUE, 0, valueLength);
        }

        // Now, we flip the phase of the input state:

        // Reverse the operations we used to construct the state:
        qftReg->X(carryIndex);
        if (partLength > 0) {
            qftReg->IndexedSBC(2 * valueLength, indexLength, valueLength, valueLength, carryIndex, toLoad);
            qftReg->X(2 * valueLength, partLength);
        }
        qftReg->IndexedSBC(2 * valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        qftReg->X(carryIndex);
        qftReg->H(partStart, 2);

        // Flip the phase of the input state at the beginning of the iteration. Only in a quaternary Grover's search,
        // we have an exact result at the end of each Grover's iteration, so we consider this an exact input for the
        // next iteration. (See the beginning of the loop, for what happens if we have more than one matching quadrant.
        qftReg->ZeroPhaseFlip(partStart, 2);
        qftReg->H(partStart, 2);
        // qftReg->PhaseFlip();
    }

    if (!foundPerm && (i == (indexLength / 2))) {
        // Here, we hit the maximum iterations, but there might be no match in the array, or there might be more than
        // one match.
        bitCapInt key = qftReg->MReg(2 * valueLength, indexLength);
        if (toLoad[key] == TARGET_VALUE) {
            foundPerm = true;
        }
    }
    if (!foundPerm && (i > 0)) {
        // If we measured an invalid value in fewer than the full iterations, or if we returned an invalid value on the
        // last iteration, we back the index up one iteration, 2 index qubits. We check the 8 boundary values. If we
        // have more than one match in the ordered list, one of our 8 boundary values is necessarily a match, since the
        // match repetitions must cross the boundary between two quadrants. If none of the 8 match, a match necessarily
        // does not exist in the ordered list.
        // This can only happen on the first iteration if the single highest and lowest values in the list cannot bound
        // the match, in which case we know a match does not exist in the list.
        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt checkIncrement = 1 << (unfixedLength - 2);
        bitCapInt key = (qftReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);
        for (i = 0; i < 4; i++) {
            // (We could either manipulate the quantum bits directly to check this, or rely on auxiliary classical
            // computing components, as need and efficiency dictate).
            if (toLoad[key | (i * checkIncrement)] == TARGET_VALUE) {
                foundPerm = true;
                qftReg->SetReg(2 * valueLength, indexLength, key | (i * checkIncrement));
                break;
            }
        }
    }

    if (!foundPerm) {
        std::cout << "Value is not in array.";
    } else {
        qftReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength, carryIndex, toLoad);
        // (If we have more than one match, this REQUIRE_THAT needs to instead check that any of the matches are
        // returned. This could be done by only requiring a match to the value register, but we want to show here that
        // the index is correct.)
        REQUIRE_THAT(qftReg,
            HasProbability(0, (2 * valueLength) + indexLength, TARGET_VALUE | (TARGET_KEY << (2 * valueLength))));
    }
    cl_free(toLoad);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_quaternary_search_alt")
{
    bitLenInt i;
    bitLenInt partStart;
    bitLenInt partLength;

    // Grover's search to find a value in an ordered list. The oracle is made with integer subtraction/addition and a
    // doubly controlled phase flip.

    const bitLenInt indexLength = 6;
    const bitLenInt valueLength = 6;
    const bitLenInt carryIndex = 19;
    const int TARGET_VALUE = 6;
    const int TARGET_KEY = 5;

    bool foundPerm = false;

    unsigned char* toLoad = cl_alloc(1 << indexLength);
    for (i = 0; i < TARGET_KEY; i++) {
        toLoad[i] = 2;
    }
    toLoad[TARGET_KEY] = TARGET_VALUE;
    for (i = (TARGET_KEY + 1); i < (1 << indexLength); i++) {
        toLoad[i] = 7;
    }

    qftReg->SetPermutation(0);
    partLength = indexLength;

    for (i = 0; i < (indexLength / 2); i++) {
        // We're in an exact permutation basis state, at this point, unless more than one quadrant contained a match for
        // our search target on the previous iteration. We can check the quadrant boundaries, without disturbing the
        // state. If there was more than one match, we either collapse into a valid state, so that we can continue as
        // expected for one matching quadrant, or we collapse into an identifiably invalid set of bounds that cannot
        // contain our match, which can be identified by checking two values and proceeding with special case logic.

        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt unfixedMask = (1 << unfixedLength) - 1;
        bitCapInt key = (qftReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);

        // (We could either manipulate the quantum bits directly to check this, or rely on auxiliary classical computing
        // components, as need and efficiency dictate).
        bitCapInt lowBound = toLoad[key];
        bitCapInt highBound = toLoad[key | unfixedMask];

        if (lowBound == TARGET_VALUE) {
            // We've found our match, and the key register already contains the correct value.
            std::cout << "Is low bound";
            foundPerm = true;
            break;
        } else if (highBound == TARGET_VALUE) {
            // We've found our match, but our key register points to the opposite bound.
            std::cout << "Is high bound";
            qftReg->X(2 * valueLength, partLength);
            foundPerm = true;
            break;
        } else if (((lowBound < TARGET_VALUE) && (highBound < TARGET_VALUE)) ||
            ((lowBound > TARGET_VALUE) && (highBound > TARGET_VALUE))) {
            // If we measure the key as a quadrant that doesn't contain our value, then either there is more than one
            // quadrant with bounds that match our target value, or there is no match to our target in the list.
            foundPerm = false;
            break;
        }

        // Prepare partial index superposition, of two most significant qubits that have not yet been fixed:
        partLength = indexLength - ((i + 1) * 2);
        partStart = (2 * valueLength) + partLength;
        qftReg->H(partStart, 2);

        // Load lower bound of quadrants:
        qftReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);

        if (partLength > 0) {
            // In this branch, our quadrant is "degenerate," (we mean, having more than one key/value pair).

            // Load upper bound of quadrants:
            qftReg->X(2 * valueLength, partLength);
            qftReg->IndexedADC(2 * valueLength, indexLength, valueLength, valueLength - 1, carryIndex, toLoad);

            // This begins the "oracle." Our "oracle" is true if the target is in this quadrant, and false otherwise:
            // Set value bits to borrow from:
            qftReg->X(valueLength - 1);
            qftReg->X(2 * valueLength - 1);
            // Subtract from the value registers with the bits to borrow from:
            qftReg->DEC(TARGET_VALUE, 0, valueLength);
            qftReg->DEC(TARGET_VALUE, valueLength, valueLength);
            // If both are higher, this is not the quadrant, and neither flips the borrow.
            // If both are lower, this is not the quadrant, and both flip the borrow.
            // If one is higher and one is lower, the low register borrow bit is flipped, and high register borrow is
            // not.
            qftReg->X(valueLength - 1);
            qftReg->CCNOT(valueLength - 1, 2 * valueLength - 1, carryIndex);
            // Flip the phase is the test bit is set:
            qftReg->Z(carryIndex);
            // Reverse everything but the phase flip:
            qftReg->CCNOT(valueLength - 1, 2 * valueLength - 1, carryIndex);
            qftReg->X(valueLength - 1);
            qftReg->INC(TARGET_VALUE, valueLength, valueLength);
            qftReg->INC(TARGET_VALUE, 0, valueLength);
            qftReg->X(2 * valueLength - 1);
            qftReg->X(valueLength - 1);
            // This ends the "oracle."
        } else {
            // In this branch, we have one key/value pair in each quadrant, so we can use our usual Grover's oracle.

            // We map from input to output.
            qftReg->DEC(TARGET_VALUE, 0, valueLength - 1);
            // Phase flip the target state.
            qftReg->ZeroPhaseFlip(0, valueLength - 1);
            // We map back from outputs to inputs.
            qftReg->INC(TARGET_VALUE, 0, valueLength - 1);
        }

        // Now, we flip the phase of the input state:

        // Reverse the operations we used to construct the state:
        qftReg->X(carryIndex);
        if (partLength > 0) {
            qftReg->IndexedSBC(2 * valueLength, indexLength, valueLength, valueLength - 1, carryIndex, toLoad);
            qftReg->X(2 * valueLength, partLength);
        }
        qftReg->IndexedSBC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);
        qftReg->X(carryIndex);
        qftReg->H(partStart, 2);

        // Flip the phase of the input state at the beginning of the iteration. Only in a quaternary Grover's search,
        // we have an exact result at the end of each Grover's iteration, so we consider this an exact input for the
        // next iteration. (See the beginning of the loop, for what happens if we have more than one matching quadrant.
        qftReg->ZeroPhaseFlip(partStart, 2);
        qftReg->H(partStart, 2);
        // qftReg->PhaseFlip();
    }

    if (!foundPerm && (i == (indexLength / 2))) {
        // Here, we hit the maximum iterations, but there might be no match in the array, or there might be more than
        // one match.
        bitCapInt key = qftReg->MReg(2 * valueLength, indexLength);
        if (toLoad[key] == TARGET_VALUE) {
            foundPerm = true;
        }
    }
    if (!foundPerm && (i > 0)) {
        // If we measured an invalid value in fewer than the full iterations, or if we returned an invalid value on the
        // last iteration, we back the index up one iteration, 2 index qubits. We check the 8 boundary values. If we
        // have more than one match in the ordered list, one of our 8 boundary values is necessarily a match, since the
        // match repetitions must cross the boundary between two quadrants. If none of the 8 match, a match necessarily
        // does not exist in the ordered list.
        // This can only happen on the first iteration if the single highest and lowest values in the list cannot bound
        // the match, in which case we know a match does not exist in the list.
        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt checkIncrement = 1 << (unfixedLength - 2);
        bitCapInt key = (qftReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);
        for (i = 0; i < 4; i++) {
            // (We could either manipulate the quantum bits directly to check this, or rely on auxiliary classical
            // computing components, as need and efficiency dictate).
            if (toLoad[key | (i * checkIncrement)] == TARGET_VALUE) {
                foundPerm = true;
                qftReg->SetReg(2 * valueLength, indexLength, key | (i * checkIncrement));
                break;
            }
        }
    }

    if (!foundPerm) {
        std::cout << "Value is not in array.";
    } else {
        qftReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);
        // (If we have more than one match, this REQUIRE_THAT needs to instead check that any of the matches are
        // returned. This could be done by only requiring a match to the value register, but we want to show here that
        // the index is correct.)
        REQUIRE_THAT(qftReg,
            HasProbability(0, (2 * valueLength) + indexLength, TARGET_VALUE | (TARGET_KEY << (2 * valueLength))));
    }
    cl_free(toLoad);
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
    cl_free(toSearch);
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtswap_bit")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->SqrtSwap(0, 1);
    qftReg->SqrtSwap(0, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sqrtswap_reg")
{
    qftReg->H(0);

    REQUIRE_FLOAT(qftReg->Prob(0), 0.5);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);

    qftReg->SqrtSwap(0, 1, 1);
    qftReg->SqrtSwap(0, 1, 1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0.5);

    qftReg->H(1);

    REQUIRE_FLOAT(qftReg->Prob(0), 0);
    REQUIRE_FLOAT(qftReg->Prob(1), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qfusion_order")
{
    QFusion optimizer(qftReg);

    optimizer.SetPermutation(0);
    optimizer.X(0);
    optimizer.CNOT(0, 1);
    optimizer.X(0);
    optimizer.CNOT(0, 1);
    optimizer.X(2);
    optimizer.CNOT(2, 3);
    optimizer.SetBit(3, false);

    optimizer.AntiCCNOT(4, 5, 6);
    optimizer.CCNOT(4, 5, 7);
    optimizer.X(6);
    optimizer.AntiCCNOT(4, 5, 6);
    optimizer.AntiCCNOT(4, 5, 7);

    qftReg = optimizer.ReleaseEngine();

    REQUIRE_THAT(qftReg, HasProbability(0, 20, 0xC6));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_timeevolve")
{
    real1 aParam = (real1)1e-4;
    real1 tDiff = 2.1f;
    real1 e0 = sqrt(ONE_R1 - aParam * aParam);

    BitOp o2neg1(new complex[4], std::default_delete<complex[]>());
    o2neg1.get()[0] = complex(e0, ZERO_R1);
    o2neg1.get()[1] = complex(-aParam, ZERO_R1);
    o2neg1.get()[2] = complex(-aParam, ZERO_R1);
    o2neg1.get()[3] = complex(e0, ZERO_R1);

    HamiltonianOpPtr h0 = std::make_shared<HamiltonianOp>(0, o2neg1);
    Hamiltonian h(1);
    h[0] = h0;

    qftReg->SetPermutation(0);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);

    bitLenInt controls[1] = { 1 };
    bool controlToggles[1] = { false };

    HamiltonianOpPtr h1 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, false, controlToggles);
    h[0] = h1;

    // The point of this "toggle" behavior is to allow enumeration of arbitrary local Hamiltonian terms with
    // permutations of a set of control bits. For example, a Hamiltonian might represent an array of local
    // electromagnetic potential wells. If there are 4 wells, each with independent potentials, control "toggles" could
    // be used on two control bits, to enumerate all four permutations of two control bits with four different local
    // Hamiltonian terms.

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);

    controlToggles[0] = true;
    HamiltonianOpPtr h2 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, false, controlToggles);
    h[0] = h2;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(qftReg->Prob(0), ZERO_R1);

    controlToggles[0] = false;
    HamiltonianOpPtr h3 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, true, controlToggles);
    h[0] = h3;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(qftReg->Prob(0), ZERO_R1);

    controlToggles[0] = true;
    HamiltonianOpPtr h4 = std::make_shared<HamiltonianOp>(controls, 1, 0, o2neg1, true, controlToggles);
    h[0] = h4;

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_timeevolve_uniform")
{
    real1 aParam = (real1)1e-4;
    real1 tDiff = 2.1f;
    real1 e0 = sqrt(ONE_R1 - aParam * aParam);

    BitOp o2neg1(new complex[8], std::default_delete<complex[]>());
    o2neg1.get()[0] = complex(ONE_R1, ZERO_R1);
    o2neg1.get()[1] = complex(ZERO_R1, ZERO_R1);
    o2neg1.get()[2] = complex(ZERO_R1, ZERO_R1);
    o2neg1.get()[3] = complex(ONE_R1, ZERO_R1);
    o2neg1.get()[4] = complex(e0, ZERO_R1);
    o2neg1.get()[5] = complex(-aParam, ZERO_R1);
    o2neg1.get()[6] = complex(-aParam, ZERO_R1);
    o2neg1.get()[7] = complex(e0, ZERO_R1);

    bitLenInt controls[1] = { 1 };

    HamiltonianOpPtr h0 = std::make_shared<UniformHamiltonianOp>(controls, 1, 0, o2neg1);
    Hamiltonian h(1);
    h[0] = h0;

    REQUIRE(h0->uniform);

    qftReg->SetPermutation(2);
    qftReg->TimeEvolve(h, tDiff);

    REQUIRE_FLOAT(abs(qftReg->Prob(0) - sin(aParam * tDiff) * sin(aParam * tDiff)), 0);
    REQUIRE_FLOAT(abs((ONE_R1 - qftReg->Prob(0)) - cos(aParam * tDiff) * cos(aParam * tDiff)), 0);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qfusion_controlled")
{
    bitLenInt controls[2] = { 1, 2 };
    real1 angles[4] = { 3.0, 0.8, 1.2, 0.7 };

    complex amps[8] = { complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };

    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 3, 0, rng);
    qftReg->SetPermutation(2);
    QInterfacePtr qftReg2 = qftReg->Clone();

    qftReg->UniformlyControlledRY(controls, 2, 0, angles);
    qftReg2->QInterface::UniformlyControlledRY(controls, 2, 0, angles);

    complex a, b;
    for (bitCapInt i = 0; i < 8; i++) {
        a = qftReg->GetAmplitude(i);
        b = qftReg2->GetAmplitude(i);
        REQUIRE_FLOAT(real(a), real(b));
        REQUIRE_FLOAT(imag(a), imag(b));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qneuron")
{
    const bitLenInt InputCount = 4;
    const bitLenInt OutputCount = 4;
    const bitCapInt InputPower = 1U << InputCount;
    const bitCapInt OutputPower = 1U << OutputCount;
    const real1 eta = 0.5;

    qftReg->Dispose(0, qftReg->GetQubitCount() - (InputCount + OutputCount));

    bitLenInt inputIndices[InputCount];
    for (bitLenInt i = 0; i < InputCount; i++) {
        inputIndices[i] = i;
    }

    std::vector<QNeuronPtr> outputLayer;
    for (bitLenInt i = 0; i < OutputCount; i++) {
        outputLayer.push_back(std::make_shared<QNeuron>(qftReg, inputIndices, InputCount, InputCount + i));
    }

    // Train the network to associate powers of 2 with their log2()
    bitCapInt perm, comp, test;
    bool bit;
    for (perm = 0; perm < InputPower; perm++) {
        comp = (~perm) + 1U;
        for (bitLenInt i = 0; i < OutputCount; i++) {
            qftReg->SetPermutation(perm);
            bit = comp & (1U << i);
            outputLayer[i]->LearnPermutation(bit, eta);
        }
    }

    for (perm = 0; perm < InputPower; perm++) {
        qftReg->SetPermutation(perm);
        for (bitLenInt i = 0; i < OutputCount; i++) {
            outputLayer[i]->Predict();
        }
        comp = qftReg->MReg(InputCount, OutputCount);
        test = ((~perm) + 1U) & (OutputPower - 1);
        REQUIRE(comp == test);
    }
}
