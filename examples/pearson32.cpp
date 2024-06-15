//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates the application of a simple hash function with Qrack.
// A quantum computer could be used to load and process many potential hashing inputs
// in superposition. Grover's search could be used to find outputs with desirable
// qualities, like a satisfied proof-of-work requirement. Such applications could
// commonly rely on the equivalent of Qrack's "Hash" function, which is demonstrated
// here. "QAlu::Hash" performs a one-to-one (unitary) transformation of a qubit
// register as a hash table key into its value.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#include <algorithm> // std::shuffle
#include <cstddef> // size_t
#include <iostream> // std::cout
#include <random> // std::default_random_engine

using namespace Qrack;

const size_t TABLE_SIZE = 256;
const size_t KEY_SIZE = 4;
const size_t HASH_SIZE = 4;

bitCapInt Pearson(const unsigned char* x, size_t len, const unsigned char* T)
{
    size_t i;
    size_t j;
    unsigned char h;
    unsigned char hh[HASH_SIZE];

    for (j = 0; j < HASH_SIZE; ++j) {
        // Change the first byte
        h = T[(x[0] + j) & 0xFF];
        for (i = 1; i < len; ++i) {
            h = T[h ^ x[i]];
        }
        hh[j] = h;
    }

    bitCapInt result = ZERO_BCI;
    for (j = 0; j < HASH_SIZE; j++) {
        bi_or_ip(&result, hh[j] << ((HASH_SIZE - (j + 1U)) * 8U));
    }

    return result;
}

void QPearson(size_t len, unsigned char* T, QAluPtr qReg)
{
    size_t i;
    size_t j;
    size_t k;
    bitLenInt x_index;
    bitLenInt h_index = (len + HASH_SIZE - 1U) * 8;

    QInterfacePtr qi = std::dynamic_pointer_cast<QInterface>(qReg);

    for (j = 0; j < HASH_SIZE; ++j) {
        // Change the first byte
        x_index = 0;
        qReg->IndexedLDA(x_index, 8, h_index, 8, T, false);
        for (i = 1; i < len; ++i) {
            x_index += 8;
            // XOR might collapse the state, as we have defined the API.
            for (k = 0; k < 8; k++) {
                qi->XOR(x_index + k, h_index + k, h_index + k);
            }
            // This is a valid API if the hash table is one-to-one (unitary).
            qReg->Hash(h_index, 8, T);
        }
        if (j < (HASH_SIZE - 1U)) {
            qReg->INC(ONE_BCI, 0, 8);
        }
        h_index -= 8;
    }
    qReg->DEC(HASH_SIZE - 2U, 0, 8);
}

int main()
{
    size_t i;

    QInterfacePtr qi = CreateQuantumInterface({ QINTERFACE_QUNIT, QINTERFACE_CPU }, 8U * (KEY_SIZE + HASH_SIZE),
        ZERO_BCI, nullptr, CMPLX_DEFAULT_ARG, true, true, false, -1, true, true);
    QAluPtr qReg = std::dynamic_pointer_cast<QAlu>(qi);

    unsigned char T[TABLE_SIZE];
    for (i = 0; i < TABLE_SIZE; i++) {
        T[i] = i;
    }
    auto rng = std::default_random_engine{};
    std::shuffle(T, T + TABLE_SIZE, rng);

    unsigned char x[KEY_SIZE];
    for (i = 0; i < KEY_SIZE; i++) {
        x[i] = (int)(256 * qi->Rand());
    }

    bitCapInt xFull = ZERO_BCI;
    for (i = 0; i < KEY_SIZE; i++) {
        bi_or_ip(&xFull, x[i] << (i * 8U));
    }
    qi->SetPermutation(xFull);
    QPearson(KEY_SIZE, T, qReg);

    bitCapInt classicalResult = Pearson(x, KEY_SIZE, T);
    bitCapInt quantumResult = qi->MReg(8 * KEY_SIZE, 8 * HASH_SIZE);

    std::cout << "Classical result: " << (bitCapIntOcl)classicalResult << std::endl;
    std::cout << "Quantum result:   " << (bitCapIntOcl)quantumResult << std::endl;

    qi->SetPermutation(ZERO_BCI);
    qi->H(0, 8);
    QPearson(KEY_SIZE, T, qReg);

    try {
        qi->ForceM(8U * KEY_SIZE, false);
    } catch (...) {
        std::cout << "Even result:      (failed)" << std::endl;
        return 0;
    }

    bitCapInt quantumKey = qi->MReg(0, 8U * KEY_SIZE);
    quantumResult = qi->MReg(8U * KEY_SIZE, 8U * HASH_SIZE);
    std::cout << "Even result:      (key: " << (bitCapIntOcl)quantumKey << ", hash: " << (bitCapIntOcl)quantumResult
              << ")" << std::endl;
};
