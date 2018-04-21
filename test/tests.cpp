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
#include "qinterface.hpp"
#include "qengine_cpu.hpp"
#include "qunit.hpp"

#include "tests.hpp"

using namespace Qrack;

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B) do {                                            \
        double __tmp_a = A;                                                 \
        double __tmp_b = B;                                                 \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                             \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                             \
    } while (0);

void print_bin(int bits, int d);
void validate_equal(QEngineCPUPtr a, QEngineCPUPtr b);
void log(QInterfacePtr p);

void print_bin(int bits, int d)
{
    int mask = 1 << bits;
    while (mask != 0) {
        printf("%d", !!(d & mask));
        mask >>= 1;
    }
}

void validate_equal(QEngineCPUPtr a, QEngineCPUPtr b)
{
    /* Validate that 'a' and 'b' are the same. */
    REQUIRE(b->GetQubitCount() == a->GetQubitCount());
    REQUIRE(b->GetMaxQPower() == a->GetMaxQPower());

    /* Test probabilities */
    for (int i = 0; i < a->GetQubitCount(); i++) {
        REQUIRE(a->Prob(i) == b->Prob(i));
    }

    /* Test the raw state vector, only valid under narrow conditions. */
    for (int i = 0; i < a->GetMaxQPower(); i++) {
        // if (a->GetState()[i]._val[0] != b->GetState()[i]._val[0] ||
        //         a->GetState()[i]._val[1] != b->GetState()[i]._val[1]) {
        //     print_bin(16, i); printf(". %f/%f != %f/%f\n", a->GetState()[i]._val[0], a->GetState()[i]._val[1],
        //         b->GetState()[i]._val[0], b->GetState()[i]._val[1]);
        // }

        REQUIRE(a->GetState()[i]._val[0] == b->GetState()[i]._val[0]);
        REQUIRE(a->GetState()[i]._val[1] == b->GetState()[i]._val[1]);
    }

}

void log(QInterfacePtr p)
{
    std::cout << std::endl << std::showpoint << p << std::endl;
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

    qengine->par_for(0, NUM_ENTRIES, [&](const bitCapInt lcv) {
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

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 1, [&](const bitCapInt lcv) {
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

    qengine->par_for_skip(0, NUM_ENTRIES, 4, 3, [&](const bitCapInt lcv) {
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

    qengine->par_for_mask(0, NUM_ENTRIES, skipArray, 2, [&](const bitCapInt lcv) {
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
        for (int i = 0; i < NUM_SKIP; i++) {
            REQUIRE((lcv & skipArray[i]) == 0);
        }
        calls++;
    });
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_superposition_reg")
{
    int j;

    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char testPage[256];
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    unsigned char expectation = qftReg->SuperposeReg8(0, 8, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x303));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_adc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(0);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0));

    qftReg->H(8, 8);
    unsigned char testPage[256];
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }

    qftReg->SuperposeReg8(8, 0, testPage);

    for (j = 0; j < 256; j++) {
        testPage[j] = 255 - j;
    }
    unsigned char expectation = qftReg->AdcSuperposeReg8(8, 0, 16, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xff));
    REQUIRE(expectation == 0xff);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_sbc_superposition_reg")
{
    int j;

    qftReg->SetPermutation(1 << 16);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 1 << 16));

    qftReg->H(8, 8);
    unsigned char testPage[256];
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    qftReg->SuperposeReg8(8, 0, testPage);

    unsigned char expectation = qftReg->SbcSuperposeReg8(8, 0, 16, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 1 << 16));
    REQUIRE(expectation == 0x00);
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_m")
{
    REQUIRE(qftReg->MReg(0, 8) == 0);
    qftReg->SetReg(0, 8, 0x2b);
    REQUIRE(qftReg->MReg(0, 8) == 0x2b);
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

TEST_CASE_METHOD(QInterfaceTestFixture, "test_incsc")
{
    int i;

    qftReg->SetPermutation(0x07f);
    for (i = 0; i < 8; i++) {
        qftReg->INCSC(1, 8, 8, 18, 19);
        REQUIRE_THAT(qftReg, HasProbability(0x07f + ((i + 1) << 8)));
    }
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_not")
{
    qftReg->SetPermutation(0x1f);
    qftReg->X(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xe0));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_swap")
{
    qftReg->SetPermutation(0xb2);
    qftReg->Swap(0, 4, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_rol")
{
    qftReg->SetPermutation(6);
    REQUIRE_THAT(qftReg, HasProbability(6));
    qftReg->ROL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(6 << 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_ror")
{
    qftReg->SetPermutation(160);
    REQUIRE_THAT(qftReg, HasProbability(160));
    qftReg->ROR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(160 >> 1));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_and")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLAND(0, 0x0f, 4, 4); // 0x0e & 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xee));
    qftReg->SetPermutation(0x3e);
    qftReg->AND(0, 4, 8, 4); // 0xe & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x23e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_or")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLOR(0, 0x0f, 4, 4); // 0x0e | 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xfe));
    qftReg->SetPermutation(0x3e);
    qftReg->OR(0, 4, 8, 4); // 0xe | 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xf3e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_xor")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLXOR(0, 0x0f, 4, 4); // 0x0e ^ 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0x1e));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 8, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_qft_h")
{
    double qftProbs[20];
    qftReg->SetPermutation(85);

    int i, j;

    // std::cout << "Quantum Fourier transform of 85 (1+4+16+64), with 1 bits first passed through Hadamard gates:"
    //           << std::endl;

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    // std::cout << "Initial:" << std::endl;
    // for (i = 0; i < 8; i++) {
    //     std::cout << "Bit " << i << ", Chance of 1:" << qftReg->Prob(i) << std::endl;
    // }

    qftReg->QFT(0, 8);

    // std::cout << "Final:" << std::endl;
    // for (i = 0; i < 8; i++) {
    //    qftProbs[i] = qftReg->Prob(i);
    //    std::cout << "Bit " << i << ", Chance of 1:" << qftProbs[i] << std::endl;
    //}

    // TODO: Without the cout statements, this provides no verification, except that the method doesn't throw an exception. 
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_decohere")
{
    int j;

    QEngineCPUPtr qftReg2 = std::make_shared<QEngineCPU>(4, 0);

    qftReg->SetPermutation(0x2b);
    qftReg->Decohere(0, 4, qftReg2);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_dispose")
{
    int j;

    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(0, 4);

    REQUIRE_THAT(qftReg, HasProbability(0, 4, 0x2));
}

TEST_CASE_METHOD(QInterfaceTestFixture, "test_cohere")
{
    int j;

    qftReg->Dispose(0, qftReg->GetQubitCount() - 4);
    qftReg->SetPermutation(0x0b);
    QEngineCPUPtr qftReg2 = std::make_shared<QEngineCPU>(4, 0x02);

    qftReg->Cohere(qftReg2);

    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
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

    unsigned char toLoad[256];
    for (i = 0; i < 256; i++) {
        toLoad[i] = 1;
    }
    toLoad[230] = 100;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(8, 8);
    qftReg->SuperposeReg8(8, 0, toLoad);

    // std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        // Our "oracle" is true for an input of "100" and false for all other inputs.
        qftReg->DEC(100, 0, 8);
        qftReg->ZeroPhaseFlip(0, 8);
        qftReg->INC(100, 0, 8);
        // This ends the "oracle."
        qftReg->X(16);
        qftReg->SbcSuperposeReg8(8, 0, 16, toLoad);
        qftReg->X(16);
        qftReg->H(8, 8);
        qftReg->ZeroPhaseFlip(8, 8);
        qftReg->H(8, 8);
        qftReg->PhaseFlip();
        qftReg->AdcSuperposeReg8(8, 0, 16, toLoad);
        // std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qftReg->ProbAll(TARGET_PROB) << std::endl;
    }

    // std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    // std::cout << "Full Result:    " << qftReg << std::endl;
    // std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
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
    unsigned char toSearch[256];

    // Create the lookup table
    for (i = 0; i < 256; i++) {
        toSearch[i] = 100;
    }

    // Divide qftReg into two registers of 8 bits each
    qftReg->SetPermutation(0);
    qftReg->H(8, 8);
    qftReg->SuperposeReg8(8, 0, toSearch);
    qftReg->H(8, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, 100));
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

/*
TEST_CASE("test_qengine_cpu_coherence_swap")
{
    // Set up four engines, identical.
    std::shared_ptr<std::default_random_engine> rng_a = std::make_shared<std::default_random_engine>();
    std::shared_ptr<std::default_random_engine> rng_b = std::make_shared<std::default_random_engine>();
    std::shared_ptr<std::default_random_engine> rng_c = std::make_shared<std::default_random_engine>();
    rng_a->seed(10);
    rng_b->seed(10);
    rng_c->seed(10);

    // 'a' is the guide.  'b' gets a reversed Cohere, and 'c' gets a normal Cohere for cross-check.
    QEngineCPUPtr a_1 = std::make_shared<QEngineCPU>(8, 0, rng_a);
    QEngineCPUPtr a_2 = std::make_shared<QEngineCPU>(8, 0, rng_a);
    QEngineCPUPtr b_1 = std::make_shared<QEngineCPU>(8, 0, rng_b);
    QEngineCPUPtr b_2 = std::make_shared<QEngineCPU>(8, 0, rng_b);
    QEngineCPUPtr c_1 = std::make_shared<QEngineCPU>(8, 0, rng_c);
    QEngineCPUPtr c_2 = std::make_shared<QEngineCPU>(8, 0, rng_c);

    // Copy the state from 'a' to 'b' and 'c'
    b_1->SetQuantumState(a_1->GetState());
    b_2->SetQuantumState(a_2->GetState());
    c_1->SetQuantumState(a_1->GetState());
    c_2->SetQuantumState(a_2->GetState());

    validate_equal(a_1, b_1);
    validate_equal(a_2, b_2);
    validate_equal(a_1, c_1);
    validate_equal(a_2, c_2);

    // Perform the same operation on 'a', 'b', and 'c'.
    a_2->H(0, 8);
    b_2->H(0, 8);
    c_2->H(0, 8);

    validate_equal(a_2, b_2);
    validate_equal(a_2, c_2);

    a_1->Cohere(a_2);       // Cohere [ reg_1, reg_2 ]
    b_2->Cohere(b_1);       // Cohere [ reg_2, reg_1 ] - backwards
    c_1->Cohere(c_2);       // Cohere [ reg_1, reg_2 ]

    QEngineCPUPtr a = a_1;
    QEngineCPUPtr b = b_2;
    QEngineCPUPtr c = c_1;

    REQUIRE(a->GetQubitCount() == 16);
    REQUIRE(b->GetQubitCount() == 16);
    REQUIRE(c->GetQubitCount() == 16);

    // Validate 'a' == 'c'.
    validate_equal(a, c);

    // 'b' is backwards, swap the first 8 bits with the second 8 bits.
    b->Swap(0, 8, 8);

    // Validate that 'a' and 'b' are the same.
    validate_equal(a, b);
}
*/

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
