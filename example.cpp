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

    qftReg->SetReg(0, 8, 0x300);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x300));

    unsigned char testPage[256];
    for (j = 0; j < 256; j++) {
        testPage[j] = j;
    }
    qftReg->SuperposeReg8(8, 0, testPage);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0x303));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_m") { REQUIRE(qftReg->MReg(0, 8) == 0); }

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_zero_flag")
{
    qftReg->SetPermutation(0);
    REQUIRE_THAT(*qftReg, HasProbability(0, 9, 0));
    qftReg->SetZeroFlag(0, 8, 8);
    REQUIRE_THAT(*qftReg, HasProbability(0, 9, 0x100));
    qftReg->SetZeroFlag(0, 8, 8);
    REQUIRE_THAT(*qftReg, HasProbability(0, 9, 0));
}*/

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

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_incsc")
{
    int i;

    qftReg->SetPermutation(0x07f);
    for (i = 0; i < 8; i++) {
        qftReg->INCSC(1, 8, 8, 18, 19);
        REQUIRE_THAT(qftReg, HasProbability(0x07f + ((i + 1) << 8)));
    }
}

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_decsc")
{
    int i;
    int start = 0x80;

    qftReg->SetPermutation(start);
    for (i = 0; i < 8; i++) {
        qftReg->DECSC(9, 0, 8, 8, 9);
        start -= 9;
        if (i == 0) {
            // First subtraction flips the flag.
            REQUIRE_THAT(qftReg, HasProbability(start | 0x100));
        } else {
            REQUIRE_THAT(qftReg, HasProbability(start));
        }
    }
}*/

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_not")
{
    qftReg->SetPermutation(31);
    qftReg->X(0, 8);
    REQUIRE_THAT(qftReg, HasProbability(0, 8, 0xe0));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_rol")
{
    qftReg->SetPermutation(160);
    REQUIRE_THAT(qftReg, HasProbability(160));
    qftReg->ROL(1, 4, 4);
    REQUIRE_THAT(qftReg, HasProbability(160 << 1));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_ror")
{
    qftReg->SetPermutation(160);
    REQUIRE_THAT(qftReg, HasProbability(160));
    qftReg->ROR(1, 4, 4);
    REQUIRE_THAT(qftReg, HasProbability(160 >> 1));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_and")
{
    std::cout << "AND Test:" << std::endl;
    qftReg->SetPermutation(0x2e);
    REQUIRE_THAT(qftReg, HasProbability(0x2e));
    qftReg->CLAND(0, 0xff, 0, 8);   // 0x2e & 0xff
    REQUIRE_THAT(qftReg, HasProbability(0x2e));
    qftReg->SetPermutation(0x3e);
    qftReg->AND(0, 4, 0, 4);        // 0xe & 0x3
    REQUIRE_THAT(qftReg, HasProbability(0x32));
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_or")
{
    std::cout << "OR Test:" << std::endl;
    qftReg->SetPermutation(38);
    std::cout << "[6,9) = [0,3) & [3,6):" << std::endl;
    WARN(qftReg);
    qftReg->CLOR(0, 255, 0, 8);
    WARN(qftReg);
    qftReg->SetPermutation(58);
    std::cout << "[0,4) = [0,4) & [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->OR(0, 4, 0, 4);
    WARN(qftReg);
}

TEST_CASE_METHOD(CoherentUnitTestFixture, "test_xor")
{
    std::cout << "XOR Test:" << std::endl;
    qftReg->SetPermutation(38);
    std::cout << "[6,9) = [0,3) & [3,6):" << std::endl;
    WARN(qftReg);
    qftReg->XOR(0, 3, 6, 3);
    WARN(qftReg);
    qftReg->SetPermutation(58);
    std::cout << "[0,4) = [0,4) & [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->XOR(0, 4, 0, 4);
    WARN(qftReg);
}

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_add")
{
    int i;

    std::cout << "ADD Test:" << std::endl;
    qftReg->SetPermutation(38);
    std::cout << "[0,4) = [0,4) + [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->ADD(0, 4, 4);
    WARN(qftReg);

    qftReg->SetPermutation(0);
    for (i = 0; i < 8; i++) {
        qftReg->H(i);
    }
    WARN(qftReg);
}*/

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_sub")
{
    std::cout << "SUB Test:" << std::endl;
    qftReg->SetPermutation(38);
    std::cout << "[0,4) = [0,4) - [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->SUB(0, 4, 4);
    WARN(qftReg);

    // qftReg->SetPermutation(0);
    // for (i = 0; i < 8; i++) {
    //	qftReg->H(i);
    //}
}*/

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_addsc")
{
    std::cout << "ADDSC Test:" << std::endl;
    qftReg->SetPermutation(55);
    // qftReg->H(0);
    // qftReg->H(8);
    std::cout << "[0,4) = [0,4) + [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->ADDSC(0, 4, 4, 8, 9);
    WARN(qftReg);

    // qftReg->SetPermutation(0);
    // for (i = 0; i < 8; i++) {
    //	qftReg->H(i);
    //}
}*/

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_subsc")
{
    int i;

    std::cout << "SUBSC Test:" << std::endl;
    qftReg->SetPermutation(56);
    std::cout << "[0,4) = [0,4) - [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->SUBSC(0, 4, 4, 8, 9);
    WARN(qftReg);

    qftReg->SetPermutation(0);
    for (i = 0; i < 8; i++) {
        qftReg->H(i);
    }
}*/

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_addbcdc")
{
    std::cout << "ADDBCDC Test:" << std::endl;
    qftReg->SetPermutation(265);
    std::cout << "[0,4) = [0,4) + [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->ADDBCDC(0, 4, 4, 8);
    WARN(qftReg);
}*/

/*TEST_CASE_METHOD(CoherentUnitTestFixture, "test_subbcdc")
{
    std::cout << "SUBBCDC Test:" << std::endl;
    qftReg->SetPermutation(256);
    std::cout << "[0,4) = [0,4) + [4,8):" << std::endl;
    WARN(qftReg);
    qftReg->SUBBCDC(0, 4, 4, 8);
    WARN(qftReg);
}*/

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

    // This matrix is ordered highest to lowest, decrementing by 1. Switch 0x64 with any other position to test.
    unsigned char toSearch[] = { 0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2,
        0xf1, 0xf0, 0xef, 0xee, 0xed, 0xec, 0xeb, 0xea, 0xe9, 0xe8, 0xe7, 0xe6, 0xe5, 0xe4, 0xe3, 0xe2, 0xe1, 0xe0,
        0xdf, 0xde, 0xdd, 0xdc, 0xdb, 0xda, 0xd9, 0xd8, 0xd7, 0xd6, 0xd5, 0xd4, 0xd3, 0xd2, 0xd1, 0xd0, 0xcf, 0xce,
        0xcd, 0xcc, 0xcb, 0xca, 0xc9, 0xc8, 0xc7, 0xc6, 0xc5, 0xc4, 0xc3, 0xc2, 0xc1, 0xc0, 0xbf, 0xbe, 0xbd, 0xbc,
        0xbb, 0xba, 0xb9, 0xb8, 0xb7, 0xb6, 0xb5, 0xb4, 0xb3, 0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xac, 0xab, 0xaa,
        0xa9, 0xa8, 0xa7, 0xa6, 0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa0, 0x9f, 0x9e, 0x9d, 0x9c, 0x9b, 0x9a, 0x99, 0x98,
        0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x91, 0x90, 0x8f, 0x8e, 0x8d, 0x8c, 0x8b, 0x8a, 0x89, 0x88, 0x87, 0x86,
        0x85, 0x84, 0x83, 0x82, 0x81, 0x80, 0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74,
        0x73, 0x72, 0x71, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62,
        0x61, 0x60, 0x5f, 0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50,
        0x4f, 0x4e, 0x4d, 0x4c, 0x4b, 0x4a, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40, 0x3f, 0x3e,
        0x3d, 0x3c, 0x3b, 0x3a, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x30, 0x2f, 0x2e, 0x2d, 0x2c,
        0x2b, 0x2a, 0x29, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x20, 0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a,
        0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
        0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00 };

    std::cout << "Grover's test:" << std::endl;
    std::cout << "(Search function is true only for an input of 100 (0x64). 100 is in position 155. First 16 bits "
                 "should output 00100110 11011001.)"
              << std::endl;
    qftReg->SetPermutation(0);

    qftReg->SetBit(16, true);
    //qftReg->H(16);
    qftReg->H(0, 8);
    // qftReg->H(8, 8);
    // qftReg->SuperposeReg8(8, 0, toSearch);

    // Literature value for Grover's should be 12, but 8 gives a higher chance of "getting lucky" on the zero bit:
    for (i = 0; i < 12; i++) {
        qftReg->DEC(100, 0, 8);
        qftReg->SetZeroFlag(0, 8, 16);
        qftReg->INC(100, 0, 8);
        qftReg->H(0, 8);
        qftReg->SetZeroFlag(0, 8, 16);
        qftReg->Z(16);
        qftReg->H(0, 8);
        std::cout << "Iteration " << i
                  << ", chance of 100:" << (qftReg->ProbAll(100) + qftReg->ProbAll(100 + (1 << 16))) << std::endl;
    }
    int greatestProbIndex = 0;
    double greatestProb = 0;
    int greatestZeroProbIndex = 0;
    double greatestZeroProb = 0;
    for (i = 0; i < 2097152; i++) {
        if (qftReg->ProbAll(i) > greatestProb) {
            greatestProb = qftReg->ProbAll(i);
            greatestProbIndex = i;
        }
    }
    std::cout << "Most likely outcome: ";
    for (i = 0; i < 21; i++) {
        if (1 << i & greatestProbIndex) {
            std::cout << "1";
        } else {
            std::cout << "0";
        }
    }
    std::cout << std::endl;
    std::cout << "Bit probabilities:" << std::endl;
    for (i = 0; i < 21; i++) {
        std::cout << "Bit " << i << ", Chance of 1:" << qftReg->Prob(i) << std::endl;
    }
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
