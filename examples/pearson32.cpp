//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This example demonstrates the application of a simple hash function with Qrack.
// A quantum computer could be used to load and process many potential hashing inputs
// in superposition. Grover's search could be used to find outputs with desirable
// qualities, like a satisfied proof-of-work requirement. Such applications could
// commonly rely on the equivalent of Qrack's "Hash" function, which is demonstrated
// here. "QInterface::Hash" performs a one-to-one (unitary) transformation of a qubit
// register as a hash table key into its value.
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

    for (j = 0; j < 4; ++j) {
        // Change the first byte
        h = T[(x[0] + j) & 0xFF];
        for (i = 1; i < len; ++i) {
            h = T[h ^ x[i]];
        }
        hh[j] = h;
    }

    return (((bitCapInt)hh[0]) << 24) | (((bitCapInt)hh[1]) << 16) | (((bitCapInt)hh[2]) << 8) | ((bitCapInt)hh[3]);
}

void QPearson32(size_t len, unsigned char* T, QInterfacePtr qReg)
{
    size_t i;
    size_t j;
    bitLenInt x_index;
    bitLenInt h_index = (len + 3) * 8;

    for (j = 0; j < 4; ++j) {
        // Change the first byte
        x_index = 0;
        qReg->IndexedLDA(x_index, 8, h_index, 8, T, false);
        for (i = 1; i < len; ++i) {
            x_index += 8;
            // XOR might collapse the state, as we have defined the API.
            qReg->XOR(x_index, h_index, h_index, 8);
            // This is a valid API if the hash table is one-to-one (unitary).
            qReg->Hash(h_index, 8, T);
        }
        if (j < 3) {
            qReg->INC(1, 0, 8);
        }
        h_index -= 8;
    }
}

int main()
{
    size_t i;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_CPU,
        32U + 8U * KEY_SIZE, 0, nullptr, CMPLX_DEFAULT_ARG, true, true, false, -1, true, true);

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
    bitCapInt quantumResult = qReg->MReg(8 * KEY_SIZE, 32);

    std::cout << "Classical result: " << (int)classicalResult << std::endl;
    std::cout << "Quantum result:   " << (int)quantumResult << std::endl;
};
