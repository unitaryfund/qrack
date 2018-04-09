#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "par_for.hpp"
#include "qregister.hpp"

#include "tests.hpp"

using namespace Qrack;

/* Begin Test Cases. */
TEST_CASE("test_par_for")
{
    size_t NUM_ENTRIES = 2000;
    std::atomic_bool hit[NUM_ENTRIES];

    for (int i = 0; i < 2000; i++) {
        hit[i].store(false);
    }

    par_for(0, NUM_ENTRIES, [&hit](const bitCapInt lcv) {
        bool old = true;
        old = hit[lcv].exchange(old);
        REQUIRE(old == false);
    });
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_superposition_reg")
{
    int j;

    qftReg->SetReg(0, 8, 0x03);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x03));

    unsigned char testPage[256];
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    testPage[0]++;
    unsigned char expectation = qftReg->SuperposeReg8(0, 8, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 16, 0x303));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_adc_superposition_reg")
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

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_sbc_superposition_reg")
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

    unsigned char expectation = qftReg->SbcSuperposeReg8(8, 0, 16, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x00));
    REQUIRE(expectation == 0x00);
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_m")
{
    qftReg->SetReg(0, 8, 0x2b);
    REQUIRE(qftReg->MReg(0, 8) == 0x2b);
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_inc")
{
    int i;

    qftReg->SetPermutation(250);
    for (i = 0; i < 8; i++) {
        qftReg->INC(1, 0, 8);
        if (i < 5) {
            REQUIRE_THAT(*qftReg, HasProbability(0, 8, 251 + i));
        } else {
            REQUIRE_THAT(*qftReg, HasProbability(0, 8, i - 5));
        }
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_dec")
{
    int i;
    int start = 0x08;

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        qftReg->DEC(9, 0, 8);
        start -= 9;
        REQUIRE_THAT(*qftReg, HasProbability(0, 19, 0xff - i * 9));
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_incc")
{
    int i;

    qftReg->SetPermutation(247 + 256);
    for (i = 0; i < 10; i++) {
        qftReg->INCC(1, 0, 8, 8);
        if (i < 7) {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 249 + i));
        } else if (i == 7) {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 0x100));
        } else {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 2 + i - 8));
        }
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_decc")
{
    int i;

    qftReg->SetPermutation(7 + 256);
    for (i = 0; i < 10; i++) {
        qftReg->DECC(1, 0, 8, 8);
        if (i < 6) {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 5 - i));
        } else if (i == 6) {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 0x1ff));
        } else {
            REQUIRE_THAT(*qftReg, HasProbability(0, 9, 253 - i + 7));
        }
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_incsc")
{
    int i;

    qftReg->SetPermutation(0x07f);
    for (i = 0; i < 8; i++) {
        qftReg->INCSC(1, 8, 8, 18, 19);
        REQUIRE_THAT(qftReg, HasProbability(0x07f + ((i + 1) << 8)));
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_not")
{
    qftReg->SetPermutation(0x1f);
    qftReg->X(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xe0));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_swap")
{
    qftReg->SetPermutation(0xb2);
    qftReg->Swap(0, 4, 4);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_rol")
{
    qftReg->SetPermutation(6);
    REQUIRE_THAT(qftReg, HasProbability(6));
    qftReg->ROL(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(6 << 1));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_ror")
{
    qftReg->SetPermutation(160);
    REQUIRE_THAT(qftReg, HasProbability(160));
    qftReg->ROR(1, 0, 8);
    REQUIRE_THAT(qftReg, HasProbability(160 >> 1));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_and")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLAND(0, 0x0f, 4, 4); // 0x0e & 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xee));
    qftReg->SetPermutation(0x3e);
    qftReg->AND(0, 4, 8, 4); // 0xe & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x23e));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_or")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLOR(0, 0x0f, 4, 4); // 0x0e | 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0xfe));
    qftReg->SetPermutation(0x3e);
    qftReg->OR(0, 4, 8, 4); // 0xe | 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xf3e));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_xor")
{
    qftReg->SetPermutation(0x0e);
    REQUIRE_THAT(qftReg, HasProbability(0x0e));
    qftReg->CLXOR(0, 0x0f, 4, 4); // 0x0e ^ 0x0f
    REQUIRE_THAT(qftReg, HasProbability(0x1e));
    qftReg->SetPermutation(0x3e);
    qftReg->XOR(0, 4, 8, 4); // 0xe ^ 0x3
    REQUIRE_THAT(qftReg, HasProbability(0xd3e));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_qft_h")
{
    double qftProbs[20];
    qftReg->SetPermutation(85);

    int i, j;

    std::cout << "Quantum Fourier transform of 85 (1+4+16+64), with 1 bits first passed through Hadamard gates:"
              << std::endl;

    for (i = 0; i < 8; i += 2) {
        qftReg->H(i);
    }

    std::cout << "Initial:" << std::endl;
    for (i = 0; i < 8; i++) {
        std::cout << "Bit " << i << ", Chance of 1:" << qftReg->Prob(i) << std::endl;
    }

    qftReg->QFT(0, 8);

    std::cout << "Final:" << std::endl;
    for (i = 0; i < 8; i++) {
        qftProbs[i] = qftReg->Prob(i);
        std::cout << "Bit " << i << ", Chance of 1:" << qftProbs[i] << std::endl;
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_decohere")
{
    int j;

    Qrack::CoherentUnit qftReg2(4, 0);

    qftReg->SetPermutation(0x2b);
    qftReg->Decohere(0, 4, qftReg2);

    REQUIRE_THAT(*qftReg, HasProbability(0, 4, 0x2));
    REQUIRE_THAT(qftReg2, HasProbability(0, 4, 0xb));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_dispose")
{
    int j;

    qftReg->SetPermutation(0x2b);
    qftReg->Dispose(0, 4);

    REQUIRE_THAT(*qftReg, HasProbability(0, 4, 0x2));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_cohere")
{
    int j;

    qftReg->Dispose(0, qftReg->GetQubitCount() - 4);
    qftReg->SetPermutation(0x0b);
    Qrack::CoherentUnit qftReg2(4, 0x02);

    qftReg->Cohere(qftReg2);

    REQUIRE_THAT(*qftReg, HasProbability(0, 8, 0x2b));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_grover")
{
    int i;

    // Grover's search inverts the function of a black box subroutine.
    // Our subroutine returns true only for an input of 100.

    const int TARGET_PROB = 100;

    // Our input to the subroutine "oracle" is 8 bits.
    qftReg->SetPermutation(0);
    qftReg->H(0, 8);

    std::cout << "Iterations:" << std::endl;
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
        std::cout << "\t" << std::setw(2) << i << "> chance of match:" << qftReg->ProbAll(TARGET_PROB) << std::endl;
    }

    std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    std::cout << "Full Result:    " << qftReg << std::endl;
    std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg(0, 8);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_set_reg")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
    qftReg->SetReg(0, 8, 10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 10));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_basis_change")
{
    int i;
    unsigned char toSearch[256];
    unsigned char output[256];

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

//("Hello, Universe!")
