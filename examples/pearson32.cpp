//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This example demonstrates the application of a simple hash function with Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <algorithm> // std::random_shuffle
#include <cstddef> // size_t
#include <iostream> // std::cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

const size_t TABLE_SIZE = 256;
const size_t KEY_SIZE = 4;

bitCapInt Pearson32(const unsigned char* x, size_t len, const unsigned char* T)
{
    size_t i;
    size_t j;
    unsigned char h;
    unsigned char hh[8];

    for (j = 0; j < 3; ++j) {
        // Change the first byte
        h = T[(x[0] + j) & 0xFF];
        for (i = 1; i < len; ++i) {
            h = T[h ^ x[i]];
        }
        hh[j] = h;
    }

    return (((bitCapInt)hh[0]) << 16) | (((bitCapInt)hh[1]) << 8) | ((bitCapInt)hh[2]);
}

void QPearson32(size_t len, unsigned char* T, QInterfacePtr qReg)
{
    size_t i;
    size_t j;
    bitLenInt x_index;
    bitLenInt h_index = (len + 2) * 8;

    for (j = 0; j < 3; ++j) {
        // Change the first byte
        x_index = 0;
        qReg->IndexedLDA(x_index, 8, h_index, 8, T, false);
        std::cout << "Loaded." << std::endl;
        for (i = 1; i < len; ++i) {
            x_index += 8;
            // XOR might collapse the state, as we have defined the API.
            qReg->XOR(x_index, h_index, h_index, 8);
            std::cout << "XOR-ed." << std::endl;
            // This is a valid API if the hash table is one-to-one (unitary).
            qReg->Hash(h_index, 8, T);
            std::cout << "Hashed." << std::endl;
        }
        if (j < 2) {
            qReg->INC(1, 0, 8);
            std::cout << "Incremented." << std::endl;
        }
        h_index -= 8;
    }
}

int main()
{
    size_t i;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_CPU,
        24U + 8U * KEY_SIZE, 0, nullptr, CMPLX_DEFAULT_ARG, true, true, false, -1, true, true);

    unsigned char T[TABLE_SIZE];
    for (i = 0; i < TABLE_SIZE; i++) {
        T[i] = i;
    }
    std::random_shuffle(T, T + TABLE_SIZE);

    unsigned char x[KEY_SIZE];
    for (i = 0; i < KEY_SIZE; i++) {
        x[i] = (int)(256 * qReg->Rand());
    }

    bitCapInt xFull = 0;
    for (i = 0; i < KEY_SIZE; i++) {
        xFull |= ((bitCapInt)x[i]) << (i * 8U);
    }
    qReg->SetPermutation(xFull);
    QPearson32(KEY_SIZE, T, qReg);

    bitCapInt classicalResult = Pearson32(x, KEY_SIZE, T);
    bitCapInt quantumResult = qReg->MReg(8 * KEY_SIZE, 24);

    std::cout << "Classical result: " << (int)classicalResult << std::endl;
    std::cout << "Quantum result:   " << (int)quantumResult << std::endl;

    qReg->SetPermutation(0);
    qReg->H(0);
    std::cout << "Initialized." << std::endl;
    QPearson32(KEY_SIZE, T, qReg);
    /*qReg->ForceM(8 * KEY_SIZE, false);

    bitCapInt quantumKey = qReg->MReg(0, 8);
    quantumResult = qReg->MReg(8 * KEY_SIZE, 24);

    for (i = 0; i < KEY_SIZE; i++) {
        x[i] = (quantumKey >> (i * 8U)) & 0xFF;
    }
    classicalResult = Pearson32(x, KEY_SIZE, T);

    std::cout << "Find even output:  ";

    if (quantumResult == classicalResult) {
        std::cout << "(input: " << (int)quantumKey << ", output: " << (int)quantumResult << std::endl;
    } else {
        std::cout << "(Failed.)" << std::endl;
    }
    */
};
