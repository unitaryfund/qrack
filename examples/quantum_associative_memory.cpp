//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This example demonstrates an example of a "quantum associative memory" network with the Qrack::QNeuron
// class. QNeuron is a type of "neuron" that can learn and predict in superposition, for general machine learning
// purposes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"
// "qneuron.hpp" defines the QNeuron class.
#include "qneuron.hpp"

using namespace Qrack;

int main()
{
    const bitLenInt InputCount = 4;
    const bitLenInt OutputCount = 4;
    const bitCapInt InputPower = 1U << InputCount;
    // const bitCapInt OutputPower = 1U << OutputCount;
    const real1 eta = 0.5;

    // QINTERFACE_OPTIMAL uses the (single-processor) OpenCL engine type, if available. Otherwise, it falls back to
    // QEngineCPU.
    QInterfacePtr qReg =
        CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, InputCount + OutputCount, 0);

    bitLenInt inputIndices[InputCount];
    for (bitLenInt i = 0; i < InputCount; i++) {
        inputIndices[i] = i;
    }

    std::vector<QNeuronPtr> outputLayer;
    for (bitLenInt i = 0; i < OutputCount; i++) {
        outputLayer.push_back(std::make_shared<QNeuron>(qReg, inputIndices, InputCount, InputCount + i));
    }

    // Train the network to associate powers of 2 with their log2()
    bitCapInt perm;
    bitCapInt comp;
    bool bit;
    std::cout << "Learning (Two's complement)..." << std::endl;
    for (perm = 0; perm < InputPower; perm++) {
        std::cout << "Epoch " << (uint64_t)(perm + 1U) << " out of " << (uint64_t)InputPower << std::endl;
        comp = (~perm) + 1U;
        for (bitLenInt i = 0; i < OutputCount; i++) {
            qReg->SetPermutation(perm);
            bit = comp & (1U << i);
            outputLayer[i]->LearnPermutation(bit, eta);
        }
    }

    std::cout << "Should associate each input with its two's complement as output..." << std::endl;
    for (perm = 0; perm < InputPower; perm++) {
        qReg->SetPermutation(perm);
        for (bitLenInt i = 0; i < OutputCount; i++) {
            outputLayer[i]->Predict();
        }
        comp = qReg->MReg(InputCount, OutputCount);
        std::cout << "Input: " << (int)perm << ", Output: " << (int)comp << std::endl;
    }
}
