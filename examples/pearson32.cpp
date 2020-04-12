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
const size_t KEY_SIZE = 2;

bitCapInt Pearson32(const unsigned char* x, size_t len, const unsigned char* T)
{
    size_t i;
    size_t j;
    unsigned char h;
    unsigned char hh[8];

    for (j = 0; j < 4; ++j) {
        // Change the first byte
        h = T[(x[0] + j) % 256];
        for (i = 1; i < len; ++i) {
            h = T[h ^ x[i]];
        }
        hh[j] = h;
    }

    return (((bitCapInt)hh[3]) << 24) | (((bitCapInt)hh[2]) << 16) | (((bitCapInt)hh[1]) << 8) | ((bitCapInt)hh[0]);
}

bitCapInt QPearson32(const unsigned char* x, size_t len, unsigned char* T, QInterfacePtr qReg)
{
    size_t i;
    size_t j;
    bitLenInt x_index;
    bitLenInt h_index = len * 8;

    bitCapInt xFull = 0;
    for (i = 0; i < len; i++) {
        xFull |= ((bitCapInt)x[i]) << (i * 8U);
    }
    qReg->SetPermutation(xFull);

    for (j = 0; j < 4; ++j) {
        // Change the first byte
        x_index = 0;
        qReg->IndexedLDA(x_index, 8, h_index, 8, T);
        for (i = 1; i < len; ++i) {
            x_index += 8;
            qReg->XOR(x_index, h_index, h_index, 8);
            // This is a valid API if the hash table is one-to-one (unitary).
            qReg->Hash(h_index, 8, T);
        }
        qReg->INC(1, 0, 8);
        h_index += 8;
    }

    return qReg->MReg(8 * len, 32);
}

int main()
{
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_CPU,
        32U + 8U * (KEY_SIZE + 1U), 0, nullptr, CMPLX_DEFAULT_ARG, true, true, false, -1, true, true);

    unsigned char T[TABLE_SIZE];
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        T[i] = i;
    }
    std::random_shuffle(T, T + TABLE_SIZE);

    unsigned char x[KEY_SIZE];
    for (size_t i = 0; i < KEY_SIZE; i++) {
        x[i] = (int)(256 * qReg->Rand());
    }

    bitCapInt classicalResult = Pearson32(x, KEY_SIZE, T);
    bitCapInt quantumResult = QPearson32(x, KEY_SIZE, T, qReg);

    std::cout << "Classical result: " << (int)classicalResult << std::endl;
    std::cout << "Quantum result:   " << (int)quantumResult << std::endl;
};
