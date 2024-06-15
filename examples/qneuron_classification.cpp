//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This example demonstrates an example of a "quantum neural classification" network with the Qrack::QNeuron
// class. QNeuron is a type of "neuron" that can learn and predict in superposition, for general machine learning
// purposes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"
// "qneuron.hpp" defines the QNeuron class.
#include "qneuron.hpp"

#include <fstream>
#include <iostream> // For cout
#include <string>

using namespace Qrack;

const bitLenInt OUTPUT_INDEX = 0;
const bitLenInt INPUT_START = 1;

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

std::vector<std::vector<BoolH>> readBinaryCSV(std::string fileName)
{
    std::vector<std::vector<BoolH>> toRet;
    std::ifstream in(fileName);
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

    dfObservation(real1_f dfValue, bool c)
    {
        df = dfValue;
        cat = c;
    }
};

void makeGeoPowerSetQnn(
    const bitLenInt& predictorCount, QInterfacePtr qReg, std::vector<QNeuronPtr>& outputLayer, std::vector<real1>& etas)
{
    bitCapIntOcl neuronCount = pow2Ocl(predictorCount);

    bitCapIntOcl i, x, y;

    std::vector<bitLenInt> allInputIndices(predictorCount);
    for (bitLenInt i = 0; i < predictorCount; i++) {
        allInputIndices[i] = INPUT_START + i;
    }

    for (i = 0; i < neuronCount; i++) {

        std::vector<bitLenInt> inputIndices;

        x = 0;
        y = i;

        while (y > 0) {
            if ((y & 1U) == 1U) {
                inputIndices.push_back(allInputIndices[x]);
            }
            x++;
            y = y >> 1U;
        }

        outputLayer.push_back(std::make_shared<QNeuron>(qReg, inputIndices, 0));
        etas.push_back((ONE_R1 / (real1)nCr(predictorCount, x)) / (real1)pow2Ocl(x));
    }
}

void train(std::vector<std::vector<BoolH>>& rawYX, std::vector<real1>& etas, QInterfacePtr qReg,
    std::vector<QNeuronPtr>& outputLayer)
{
    // Train the network
    size_t i;
    size_t rowCount = rawYX.size();
    bitLenInt qRegSize = qReg->GetQubitCount();
    bitCapIntOcl perm;
    std::vector<bitLenInt> permH;

    std::cout << "Learning..." << std::endl;

    for (size_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
        std::cout << "Epoch " << (rowIndex + 1U) << " out of " << rowCount << std::endl;

        std::vector<BoolH> row = rawYX[rowIndex];

        perm = 0U;
        permH.clear();
        for (i = 0; i < qRegSize; i++) {
            if (row[i] == BOOLH_T) {
                perm |= pow2Ocl(i);
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
                outputLayer[i]->LearnPermutation((real1_f)(etas[i] / rowCount), row[0]);
            }
        } else {
            for (i = 0; i < outputLayer.size(); i++) {
                outputLayer[i]->Learn((real1_f)(etas[i] / (rowCount * pow2Ocl(permH.size()))), row[0]);
            }
        }
    }
    std::cout << std::endl;
}

std::vector<dfObservation> predict(
    std::vector<std::vector<BoolH>>& rawYX, QInterfacePtr qReg, std::vector<QNeuronPtr>& outputLayer)
{
    // Train the network
    size_t i, rowIndex;
    size_t rowCount = rawYX.size();
    bitLenInt qRegSize = qReg->GetQubitCount();
    bitCapIntOcl perm;
    std::vector<bitLenInt> permH;

    std::vector<dfObservation> dfObs;

    for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
        std::vector<BoolH> row = rawYX[rowIndex];

        perm = 0U;
        permH.clear();
        for (i = 0; i < qRegSize; i++) {
            if (row[i] == BOOLH_T) {
                perm |= pow2Ocl(i);
            } else if (row[i] == BOOLH_H) {
                permH.push_back(i);
            }
        }

        qReg->SetPermutation(perm);
        for (i = 0; i < permH.size(); i++) {
            qReg->H(permH[i]);
        }

        for (bitCapIntOcl i = 0; i < outputLayer.size(); i++) {
            outputLayer[i]->Predict();
        }

        // Dependent variable (true classifications) must not have "H" ("NA") values
        dfObs.emplace_back(dfObservation(qReg->Prob(OUTPUT_INDEX), (row[0] == BOOLH_T)));

        std::cout << "Input: " << (perm >> 1U) << ", Output (df): " << dfObs[rowIndex].df << std::endl;
    }
    std::cout << std::endl;

    return dfObs;
}

real1_f calculateAuc(std::vector<std::vector<BoolH>>& rawYX, std::vector<dfObservation>& dfObs)
{
    size_t rowCount = rawYX.size();
    size_t rowIndex;
    size_t tp = 0, fp = 0, fn = 0, tn = 0;
    size_t oTp = 0, oFp = 0, oFn = 0, oTn = 0;
    size_t totT, totF;
    real1_f lTp = 0, lFp = 0;
    real1_f dTp, dFp;
    real1_f optimumCutoff = 0;
    real1_f cutoff;
    real1_f err, optimumErr;
    real1_f auc = 0;

    std::set<real1> dfVals;
    for (rowIndex = 0; rowIndex < dfObs.size(); rowIndex++) {
        dfVals.insert(dfObs[rowIndex].df);
    }

    for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
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
    err = ((real1_f)fp) / rowCount;
    optimumErr = err;

    std::set<real1>::iterator it = dfVals.begin();
    while (it != dfVals.end()) {
        cutoff = (real1_f)*it;
        it++;

        lTp = (real1_f)tp / totT;
        lFp = (real1_f)fp / totF;

        tp = 0;
        fp = 0;
        fn = 0;
        tn = 0;

        for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
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

        dTp = lTp - ((real1_f)tp / totT);
        dFp = lFp - ((real1_f)fp / totF);
        auc += dFp * (((real1_f)tp / totT) + (dTp / 2));

        err = ((real1_f)((fp * fp) + (fn * fn))) / (rowCount * rowCount);
        if (err < optimumErr) {
            optimumErr = err;

            if (it == dfVals.end()) {
                optimumCutoff = 1;
            } else {
                optimumCutoff = cutoff + ((((real1_f)*it) - cutoff) / 2);
            }

            oTp = tp;
            oFp = fp;
            oFn = fn;
            oTn = tn;
        }
    }

    std::cout << "AUC: " << auc << std::endl;
    std::cout << "Optimal df cutoff:" << optimumCutoff << std::endl;
    std::cout << "Confusion matrix values: " << std::endl;
    std::cout << "TP: " << oTp << std::endl;
    std::cout << "FP: " << oFp << std::endl;
    std::cout << "FN: " << oFn << std::endl;
    std::cout << "TN: " << oTn << std::endl;

    return auc;
}

int main()
{
    std::vector<std::vector<BoolH>> rawYX = readBinaryCSV("data/powers_of_2.csv");
    std::cout << "Row count: " << rawYX.size() << std::endl;
    std::cout << "Column count: " << rawYX[0].size() << std::endl;
    bitLenInt predictorCount = rawYX[0].size() - 1U;

    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, predictorCount + 1U, ZERO_BCI);

    std::vector<QNeuronPtr> outputLayer;
    std::vector<real1> etas;
    makeGeoPowerSetQnn(predictorCount, qReg, outputLayer, etas);

    train(rawYX, etas, qReg, outputLayer);

    std::vector<dfObservation> dfObs = predict(rawYX, qReg, outputLayer);

    calculateAuc(rawYX, dfObs);
}
