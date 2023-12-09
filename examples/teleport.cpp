//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates the quantum teleportation protocol, including several different ways to conceptualize of
// it.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

#include <algorithm> // std::random_shuffle
#include <cstddef> // size_t
#include <iostream> // std::cout

using namespace Qrack;

void PrintBit(QInterfacePtr qReg, bitLenInt qubit)
{
    std::cout << "Z Prob.=" << qReg->Prob(qubit) << std::endl;
    qReg->H(qubit);
    std::cout << "X Prob.=" << qReg->Prob(qubit) << std::endl;
    qReg->H(qubit);
}

void StatePrep(QInterfacePtr qReg)
{
    // "Alice" has a qubit to teleport.

    // (To test consistency, comment out this U() gate.)
    real1_f theta = (real1_f)(2 * PI_R1 * qReg->Rand());
    real1_f phi = (real1_f)(2 * PI_R1 * qReg->Rand());
    real1_f lambda = (real1_f)(2 * PI_R1 * qReg->Rand());
    qReg->U(0, theta, phi, lambda);

    // (Try with and without just an X() gate, instead.)
    // qReg->X(0);

    std::cout << "Alice is sending:" << std::endl;
    PrintBit(qReg, 0U);
}

int main()
{
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 3, ZERO_BCI);
    // "Eve" prepares a Bell pair.
    qReg->H(1);
    qReg->CNOT(1, 2);
    // Alice prepares her "message."
    StatePrep(qReg);
    // Alice entangles her message with her half of the Bell pair.
    qReg->CNOT(0, 1);
    qReg->H(0);
    // Alice measures both of her bits
    bool q0 = qReg->M(0);
    bool q1 = qReg->M(1);
    // "Bob" receives classical message and prepares his half of the Bell pair to complete teleportation.
    if (q0) {
        qReg->Z(2);
    }
    if (q1) {
        qReg->X(2);
    }
    std::cout << "Bob received:" << std::endl;
    PrintBit(qReg, 2U);

    // MWI unitary equivalent:
    qReg->SetPermutation(ZERO_BCI);
    // Eve prepares a Bell pair.
    qReg->H(1);
    qReg->CNOT(1, 2);
    // Alice prepares her message.
    StatePrep(qReg);
    // Alice entangles her message with her half of the Bell pair.
    qReg->CNOT(0, 1);
    qReg->H(0);
    // Alice measures both of her bits
    // Bob receives the classical message and prepares his half of the Bell pair to complete teleportation.
    qReg->CZ(0, 2);
    qReg->CNOT(1, 2);
    std::cout << "Bob received:" << std::endl;
    PrintBit(qReg, 2U);

    // Another MWI unitary equivalent, with a caveat: This variant would specifically be "decoherent," if measurements
    // were used instead of unitary gates.
    qReg->SetPermutation(ZERO_BCI);
    // Eve prepares a Bell pair.
    qReg->H(1);
    qReg->CNOT(1, 2);
    // Alice prepares her message.
    StatePrep(qReg);
    // Alice entangles her message with her half of the Bell pair.
    qReg->CNOT(0, 1);
    qReg->H(0);
    // Alice measures her message bit.
    // Bob receives the classical message and starts to prepares his half of the Bell pair.
    qReg->CZ(0, 2);
    // Alice and Bob "reverse control"
    qReg->H(1);
    qReg->H(2);
    // Bob measures his half of the Bell pair and sends the result to Alice.
    // Alice receives the classical message and prepares her half of the Bell pair to complete teleportation.
    qReg->CNOT(2, 1);
    // Alice and Bob "reverse control" again.
    qReg->H(1);
    qReg->H(2);
    std::cout << "Bob received:" << std::endl;
    PrintBit(qReg, 2U);
};
