//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.


#pragma once

#include "qinterface.hpp"

namespace Qrack {

class QNeuron;
typedef std::shared_ptr<QNeuron> QNeuronPtr;

class QNeuron {
private:
    QInterfacePtr qReg;
    bitLenInt* inputIndices;
    bitLenInt inputCount;
    bitCapInt inputPower;
    bitLenInt outputIndex;
    real1* angles;

    const real1 tolerance = 1e-12;

public:
    /** "Quantum neuron" or "quantum perceptron" class that can learn and predict in superposition
      *
      * This is a simple "quantum neuron" or "quantum perceptron" class, for use of the Qrack library for machine learning. See https://arxiv.org/abs/1711.11240 for the basis of this class' theoretical concept. (That paper does not use the term "uniformly controlled rotation gate," but "conditioning on all controls" is computationally the same.) */
    QNeuron(QInterfacePtr reg, bitLenInt* inputIndcs, bitLenInt inputCnt, bitLenInt outputIndx)
        : inputCount(inputCnt)
        , inputPower(1U << inputCnt)
        , outputIndex(outputIndx)
    {
        qReg = reg;

        inputIndices = new bitLenInt[inputCount];
        std::copy(inputIndcs, inputIndcs + inputCount, inputIndices);
        
        angles = new real1[inputPower];
        std::fill(angles, angles + inputPower, ZERO_R1);
    }

    /** Create a new QNeuron which is an exact duplicate of another, including its learned state. */
    QNeuron(const QNeuron& toCopy)
        : QNeuron(toCopy.qReg, toCopy.inputIndices, toCopy.inputCount, toCopy.outputIndex)
    {
        std::copy(toCopy.angles, toCopy.angles + inputPower, angles);
    }

    ~QNeuron()
    {
        delete[] inputIndices;
        delete[] angles;
    }

    /** Feed-forward from the inputs, loaded in "qReg", to a binary categorical distinction. "expected" flips the binary categories, if false. */
    real1 Predict(bool expected = true)
    {
        qReg->SetBit(outputIndex, false);
        qReg->UniformlyControlledRY(inputIndices, inputCount, outputIndex, angles);
        real1 prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1 - prob;
        }
        return prob;
    }

    /** Perform one learning iteration
      *
      * Inputs must be already loaded into "qReg" before calling this method. "expected" is the true binary output category, for training. "eta" is a volatility or "learning rate" parameter with a maximum value of 1.
      */ 
    void Learn(bool expected, real1 eta)
    {
        real1 startProb, endProb;

        startProb = Predict(expected);
        if (startProb > (ONE_R1 - tolerance)) {
            return;
        }

        for (bitCapInt perm = 0; perm < inputPower; perm++)
        {
            angles[perm] += eta * M_PI;

            endProb = Predict(expected);
            if (endProb > (ONE_R1 - tolerance)) {
                return;
            }

            if (endProb > startProb) {
                startProb = endProb;
            } else {
                angles[perm] -= 2 * eta * M_PI;

                endProb = Predict(expected);
                if (endProb > (ONE_R1 - tolerance)) {
                    return;
                }

                if (endProb > startProb) {
                    startProb = endProb;
                } else {
                    angles[perm] += eta * M_PI;
                }
            }
        }
    }
};
} // namespace Qrack
