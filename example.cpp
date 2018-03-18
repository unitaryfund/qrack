#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"
#include "qregister.hpp"

#include "tests.hpp"

using namespace Qrack;

/* Begin Test Cases. */

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_set_reg")
{
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0));
    qftReg->SetReg(0, 8, 10);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 10));
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

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_m") { REQUIRE(qftReg->MReg(0, 8) == 0); }

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
    qftReg->SetPermutation(31);
    qftReg->X(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xe0));
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
    std::cout << "AND Test:" << std::endl;
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

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_m_reg")
{
    int j;

    std::cout << "M Test:" << std::endl;
    std::cout << "Initial:" << std::endl;
    for (j = 0; j < 8; j++) {
        std::cout << "Bit " << j << ", Chance of 1:" << qftReg->Prob(j) << std::endl;
    }

    qftReg->M(0);
    std::cout << "Final:" << std::endl;
    for (j = 0; j < 8; j++) {
        std::cout << "Bit " << j << ", Chance of 1:" << qftReg->Prob(j) << std::endl;
    }
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

    std::cout << "Decohere test:" << std::endl;

    Qrack::CoherentUnit qftReg2(4, 0);

    qftReg->Decohere(0, 4, qftReg2);

    for (j = 0; j < 4; j++) {
        std::cout << "Bit " << j << ", Chance of 1:" << qftReg->Prob(j) << std::endl;
    }

    for (j = 0; j < 4; j++) {
        std::cout << "Bit " << (j + 4) << ", Chance of 1:" << qftReg2.Prob(j) << std::endl;
    }
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_grover")
{
    int i;

    // Search function is true only for an input of 100 (0x64). 100 is in
    // position 155. First 16 bits should output 00100110 11011001.
    //
    // The target probability here is looking for 100 in the low 8 bits, and
    // expecting to find 155 in the upper register.
    const int TARGET_PROB = 100 + (155 << 8);

    // Create a table with decreasing values to search through
    unsigned char toSearch[256];

    // Create the lookup table
    for (i = 0; i < 256; i++) {
        toSearch[i] = 255 - i;
    }

    // Make sure the value being searched for is in the targetted location.
    REQUIRE(toSearch[155] == 100);

    // Divide qftReg into two registers of 8 bits each
    qftReg->SetPermutation(0);
    qftReg->SetBit(16, true);
    qftReg->H(8, 8);
    qftReg->SuperposeReg8(8, 0, toSearch);

    std::cout << "Iterations:" << std::endl;
    // Twelve iterations maximizes the probablity for 256 searched elements.
    for (i = 0; i < 12; i++) {
        qftReg->DEC(100, 0, 8);
        qftReg->SetZeroFlag(0, 8, 16);
        qftReg->INC(100, 0, 8);
        qftReg->EntangledH(8, 0, 8);
        qftReg->SetZeroFlag(8, 8, 16);
        qftReg->EntangledH(8, 0, 8);
        qftReg->Z(16);
        std::cout << "\t" << std::setw(2) << i
                  << "> chance of match:" << (qftReg->ProbAll(TARGET_PROB) + qftReg->ProbAll(TARGET_PROB + (1 << 16)))
                  << std::endl;
    }

    std::cout << "Ind Result:     " << std::showbase << qftReg << std::endl;
    std::cout << "Full Result:    " << qftReg << std::endl;
    std::cout << "Per Bit Result: " << std::showpoint << qftReg << std::endl;

    qftReg->MReg8(0);
    qftReg->SetBit(16, false);

    REQUIRE_THAT(qftReg, HasProbability(0, 16, TARGET_PROB));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_random_walk")
{
    const int planckTimes = 65500;
    const int mpPowerOfTwo = 16;
    const int maxTrials = 1000;

    int i, j;

    std::cout << "Next step might take a while..." << std::endl;

    Qrack::CoherentUnit qReg(mpPowerOfTwo, 0);

    // 50/50 Superposition between "step" and "don't step" at each discrete time step
    // is equivalent to Pi/4 rotation around the y-axis of spin, each time:
    double angle = -M_PI / 4.0;
    // This is our starting distance from the destination point (plus one).
    unsigned int power = 1 << mpPowerOfTwo;

    // We will store our ultimate position in this variable and return it from the operation:
    // unsigned int toReturn = 0;

    double* zeroProbs = new double[mpPowerOfTwo];

    // This isn't exactly the same as a classical unidirectional discrete random walk.
    // At each step, we superpose our current distance from the destination with a distance
    // that is one unit shorter. This is equivalent to a rotation around the y-axis,
    //"Ry(Pi/4, qubit[0])", where qubit[0] is the least significant qubit of our distance.
    // Four successive steps of superposition then give a rotation of Pi.
    // A rotation of Pi about the y-axis MUST turn a pure state of |0> into a pure state of |1>
    // and vice versa.
    // Hence, we already know a maximum amount of time steps this could take, "power * 4".
    // We can just return zero if our time step count is greater than or equal to this.
    if (planckTimes / 4 < power) {
        // If we haven't exceeded the known maximum time, we carry out the algorithm.
        // We grab enough qubits and set them to the initial state.
        // Weirdly, we use |0> to represent 1 and |1> to represent 0,
        // just so we can avoid many unnecessary "not" gates, "X(...)" operations.

        // double testProb[power];
        // double totalProb;
        unsigned int workingPower = 1;
        for (i = 1; i <= planckTimes; i++) {
            // For each time step, first increment superposition in the least significant bit:
            qReg.RY(angle, 0);
            // At 2 steps, we could have a change in the second least significant bit.
            // At 4 steps, we could have a change in the third least significant bit AND the second least.
            // At 8 steps, we could have a change in the fourth least, third least, and second least.
            //(...Etc.)
            workingPower = 1;
            for (j = 1; j < mpPowerOfTwo; j++) {
                workingPower = workingPower << 1;
                if (i % workingPower == 0) {
                    //"CNOT" is a quantum "controlled not" gate.
                    // If the control bit is in a 50/50 superposition, for example,
                    // the other input bit ends up in 50/50 superposition of being reversed, or "not-ed".
                    // The two input bits can become entangled in this process! If the control bit were next
                    // immediately measured in the |1> state, we'd know the other input qubit was flipped.
                    // If the control bit was next immediately measured in the |0> state, we'd know the other input
                    // qubit was not flipped.

                    //(Here's where we avoid two unnecessary "not" or "X(...)" gates by flipping our 0/1 convention:)
                    qReg.CNOT(j - 1, j);
                }
            }

            // qReg.ProbArray(testProb);
            // totalProb = 0.0;
            // for (j = 0; j < power; j++) {
            //	totalProb += testProb[j];
            //}
            // if (totalProb < 0.999 || totalProb > 1.001) {
            //	for (j = 0; j < power; j++) {
            //		std::cout<<j<<" Prob:"<<testProb[j]<<std::endl;
            //	}
            //	std::cout<<"Total Prob is"<<totalProb<<" at iteration "<<i<<"."<<std::endl;
            //	std::cin >> testKey;
            //}
        }

        // The qubits are now in their fully simulated, superposed and entangled end state.
        // Ultimately, we have to measure a final state in the |0>/|1> basis for each bit, breaking the
        // superpositions and entanglements. Quantum measurement is nondeterministic and introduces randomness,
        // so we repeat the simulation many times in the driver code and average the results.

        for (j = 0; j < mpPowerOfTwo; j++) {
            zeroProbs[j] = 1.0 - qReg.Prob(j);
            std::cout << "Bit " << j << ", Chance of 0:" << zeroProbs[j] << std::endl;
        }
    }

    unsigned int outcome;
    unsigned int* masses = new unsigned int[1000];
    double totalMass = 0;
    for (i = 0; i < maxTrials; i++) {
        outcome = 0;
        for (j = 0; j < mpPowerOfTwo; j++) {
            if (qReg.Rand() < zeroProbs[j]) {
                outcome += 1 << j;
            }
        }
        masses[i] = outcome;
        totalMass += outcome;
    }

    delete[] zeroProbs;

    double averageMass = totalMass / maxTrials;
    double sqrDiff = 0.0;
    double diff;
    // Calculate the standard deviation of the simulation trials:
    for (int trial = 0; trial < maxTrials; trial++) {
        diff = masses[trial] - averageMass;
        sqrDiff += diff * diff;
    }
    double stdDev = sqrt(sqrDiff / (maxTrials - 1));

    std::cout << "Trials:" << maxTrials << std::endl;
    std::cout << "Starting Point:" << ((1 << mpPowerOfTwo) - 1) << std::endl;
    std::cout << "Time units passed:" << planckTimes << std::endl;
    std::cout << "Average distance left:" << averageMass << std::endl;
    std::cout << "Distance left std. dev.:" << stdDev << std::endl;

    //("Hello, Universe!")
}
