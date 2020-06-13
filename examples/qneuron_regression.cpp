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

#include <fstream>
#include <iostream> // For cout
#include <string>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"
// "qneuron.hpp" defines the QNeuron class.
#include "qneuron.hpp"

using namespace Qrack;

std::vector<std::vector<bool>> readBinaryCSV()
{
    std::vector<std::vector<bool>> toRet;
    std::ifstream in("data/powers_of_2.csv");
    std::string str;

    while (std::getline(in, str)) {
        std::vector<bool> row;
        while (str != "") {
            std::string::size_type pos = str.find(',');
            if (pos != std::string::npos) {
                row.push_back(str.substr(0, pos) == "1");
                str = str.substr(pos + 1, str.size() - pos);
            } else {
                row.push_back(str == "1");
                str = "";
            }
        }
        toRet.push_back(row);
    }

    return toRet;
}

// to find factorial
int factorial(int n)
{
    int fact = 1;
    for (int i = 1; i <= n; i++)
        fact *= i;
    return fact;
}

// to find combination or nCr
int nCr(int n, int r) { return (factorial(n) / (factorial(n - r) * factorial(r))); }

int main()
{
    std::vector<std::vector<bool>> rawYX = readBinaryCSV();
    const bitLenInt OUTPUT_INDEX = 0;
    const bitLenInt INPUT_START = 1;
    size_t trainingRowCount = rawYX.size();
    bitLenInt predictorCount = rawYX[0].size() - 1U;
    bitLenInt neuronCount = pow2(predictorCount);
    bitLenInt qRegSize = predictorCount + 1U;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, qRegSize, 0);

    std::vector<bitLenInt> allInputIndices(predictorCount);
    for (bitLenInt i = 0; i < predictorCount; i++) {
        allInputIndices[i] = INPUT_START + i;
    }

    bitLenInt i, x, y;

    std::vector<QNeuronPtr> outputLayer;
    std::vector<real1> etas;
    for (i = 0; i < neuronCount; i++) {

        std::vector<bitLenInt> inputIndices;

        x = 0;
        y = i;

        while (y > 0) {
            if ((y & 1U) == 1U) {
                inputIndices.push_back(allInputIndices[x]);
                x++;
            }
            y = y >> 1U;
        }

        outputLayer.push_back(std::make_shared<QNeuron>(qReg, &(inputIndices[0]), x, 0));
        etas.push_back((ONE_R1 / nCr(predictorCount, x)) / pow2(x));
    }

    // Train the network to recognize powers 2 (via the CSV data)
    bitCapInt perm;
    std::cout << "Learning (powers of two)..." << std::endl;
    for (size_t rowIndex = 0; rowIndex < trainingRowCount; rowIndex++) {
        std::cout << "Epoch " << (rowIndex + 1U) << " out of " << trainingRowCount << std::endl;

        std::vector<bool> row = rawYX[rowIndex];

        perm = 0U;
        for (i = 0; i < qRegSize; i++) {
            if (row[i]) {
                perm |= pow2(i);
            }
        }

        qReg->SetPermutation(perm);

        for (i = 0; i < outputLayer.size(); i++) {
            outputLayer[i]->LearnPermutation(row[0], etas[i]);
        }
    }

    std::cout << "Should identify powers of 2 (via discriminant function)..." << std::endl;
    for (perm = 0; perm < trainingRowCount; perm++) {
        qReg->SetPermutation(perm << 1U);
        for (bitLenInt i = 0; i < outputLayer.size(); i++) {
            outputLayer[i]->Predict();
        }
        std::cout << "Input: " << perm << ", Output (df): " << qReg->Prob(OUTPUT_INDEX) << std::endl;
    }
}
