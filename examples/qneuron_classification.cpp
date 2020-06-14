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

enum BoolH { BOOLH_F = 0, BOOLH_T = 1, BOOLH_H = 2 };

BoolH translateCsvEntry(std::string str)
{
    switch (str.at(0)) {
    case '0':
        return BOOLH_F;
        break;
    case '1':
        return BOOLH_T;
        break;
    case 'H':
        return BOOLH_H;
        break;
    default:
        throw "Invalid CSV character";
    }
}

std::vector<std::vector<BoolH>> readBinaryCSV()
{
    std::vector<std::vector<BoolH>> toRet;
    std::ifstream in("data/powers_of_2.csv");
    std::string str;

    while (std::getline(in, str)) {
        std::vector<BoolH> row;
        while (str != "") {
            std::string::size_type pos = str.find(',');
            row.push_back(translateCsvEntry(str));
            if (pos != std::string::npos) {
                str = str.substr(pos + 1, str.size() - pos);
            } else {
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

struct dfObservation {
    real1 df;
    bool cat;

    dfObservation(real1 dfValue, bool c)
    {
        df = dfValue;
        cat = c;
    }
};

int main()
{
    std::vector<std::vector<BoolH>> rawYX = readBinaryCSV();
    const bitLenInt OUTPUT_INDEX = 0;
    const bitLenInt INPUT_START = 1;
    size_t trainingRowCount = rawYX.size();
    bitLenInt predictorCount = rawYX[0].size() - 1U;
    bitLenInt neuronCount = pow2(predictorCount);
    bitLenInt qRegSize = predictorCount + 1U;
    bitLenInt i, x, y;
    size_t rowIndex;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_OPTIMAL, qRegSize, 0);

    std::vector<bitLenInt> allInputIndices(predictorCount);
    for (bitLenInt i = 0; i < predictorCount; i++) {
        allInputIndices[i] = INPUT_START + i;
    }

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
    std::vector<bitLenInt> permH;
    std::cout << "Learning (powers of two)..." << std::endl;
    for (rowIndex = 0; rowIndex < trainingRowCount; rowIndex++) {
        std::cout << "Epoch " << (rowIndex + 1U) << " out of " << trainingRowCount << std::endl;

        std::vector<BoolH> row = rawYX[rowIndex];

        perm = 0U;
        permH.clear();
        for (i = 0; i < qRegSize; i++) {
            if (row[i] == BOOLH_T) {
                perm |= pow2(i);
            } else if (row[i] == BOOLH_H) {
                permH.push_back(i);
            }
        }

        qReg->SetPermutation(perm);
        for (i = 0; i < permH.size(); i++) {
            qReg->H(permH[i]);
        }

        if (permH.size() == 0) {
            for (i = 0; i < outputLayer.size(); i++) {
                outputLayer[i]->LearnPermutation(row[0], etas[i] / trainingRowCount);
            }
        } else {
            for (i = 0; i < outputLayer.size(); i++) {
                outputLayer[i]->Learn(row[0], etas[i] / trainingRowCount);
            }
        }
    }

    std::vector<dfObservation> dfObs;
    std::set<real1> dfVals;
    std::cout << std::endl
              << "Should identify powers of 2 (via discriminant function, with some missing values)..." << std::endl;
    for (rowIndex = 0; rowIndex < trainingRowCount; rowIndex++) {
        std::vector<BoolH> row = rawYX[rowIndex];

        perm = 0U;
        for (i = 0; i < qRegSize; i++) {
            if (row[i]) {
                perm |= pow2(i);
            }
        }

        qReg->SetPermutation(perm);

        for (bitLenInt i = 0; i < outputLayer.size(); i++) {
            outputLayer[i]->Predict();
        }

        // Dependent variable (true classifications) must not have "H" ("NA") values
        dfObs.emplace_back(dfObservation(qReg->Prob(OUTPUT_INDEX), (row[0] == BOOLH_T)));
        dfVals.insert(dfObs[rowIndex].df);

        std::cout << "Input: " << (perm >> 1U) << ", Output (df): " << dfObs[rowIndex].df << std::endl;
    }

    size_t tp = 0, fp = 0, fn = 0, tn = 0;
    size_t oTp = 0, oFp = 0, oFn = 0, oTn = 0;
    size_t totT, totF;
    real1 lTp = 0, lFp = 0;
    real1 dTp, dFp;
    real1 optimumCutoff = 0;
    real1 cutoff;
    real1 err, optimumErr;
    real1 auc = 0;

    for (rowIndex = 0; rowIndex < trainingRowCount; rowIndex++) {
        if (dfObs[rowIndex].cat) {
            tp++;
        } else {
            fp++;
        }
    }

    oTp = tp;
    oFp = fp;
    totT = tp;
    totF = fp;
    err = ((real1)fp) / trainingRowCount;
    optimumErr = err;

    std::set<real1>::iterator it = dfVals.begin();
    while (it != dfVals.end()) {
        cutoff = *it;
        it++;

        lTp = (real1)tp / totT;
        lFp = (real1)fp / totF;

        tp = 0;
        fp = 0;
        fn = 0;
        tn = 0;

        for (rowIndex = 0; rowIndex < trainingRowCount; rowIndex++) {
            if (dfObs[rowIndex].cat) {
                if (dfObs[rowIndex].df > cutoff) {
                    tp++;
                } else {
                    fn++;
                }
            } else {
                if (dfObs[rowIndex].df > cutoff) {
                    fp++;
                } else {
                    tn++;
                }
            }
        }

        dTp = lTp - ((real1)tp / totT);
        dFp = lFp - ((real1)fp / totF);
        auc += dFp * (((real1)tp / totT) + (dTp / 2));

        err = ((real1)((fp * fp) + (fn * fn))) / (trainingRowCount * trainingRowCount);
        if (err < optimumErr) {
            optimumErr = err;

            if (it == dfVals.end()) {
                optimumCutoff = 1;
            } else {
                optimumCutoff = cutoff + (((*it) - cutoff) / 2);
            }

            oTp = tp;
            oFp = fp;
            oFn = fn;
            oTn = tn;
        }
    }

    std::cout << std::endl;
    std::cout << "AUC: " << auc << std::endl;
    std::cout << "Optimal df cutoff:" << optimumCutoff << std::endl;
    std::cout << "Confusion matrix values: " << std::endl;
    std::cout << "TP: " << oTp << std::endl;
    std::cout << "FP: " << oFp << std::endl;
    std::cout << "FN: " << oFn << std::endl;
    std::cout << "TN: " << oTn << std::endl;
}
