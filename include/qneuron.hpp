//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
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
    bitLenInt inputCount;
    bitCapIntOcl inputPower;
    bitLenInt outputIndex;
    real1 tolerance;
    std::unique_ptr<bitLenInt> inputIndices;
    std::unique_ptr<real1> angles;
    QInterfacePtr qReg;

public:
    /** "Quantum neuron" or "quantum perceptron" class that can learn and predict in superposition
     *
     * This is a simple "quantum neuron" or "quantum perceptron" class, for use of the Qrack library for machine
     * learning. See https://arxiv.org/abs/1711.11240 for the basis of this class' theoretical concept. (That paper does
     * not use the term "uniformly controlled rotation gate," but "conditioning on all controls" is computationally the
     * same.)
     *
     * An untrained QNeuron (with all 0 variational parameters) will forward all inputs to 1/sqrt(2) * (|0> + |1>). The
     * variational parameters are Pauli Y-axis rotation angles divided by 2 * Pi (such that a learning parameter of 0.5
     * will train from a default output of 0.5/0.5 probability to either 1.0 or 0.0 on one training input). */
    QNeuron(
        QInterfacePtr reg, bitLenInt* inputIndcs, bitLenInt inputCnt, bitLenInt outputIndx, real1_f tol = REAL1_EPSILON)
        : inputCount(inputCnt)
        , inputPower(pow2Ocl(inputCnt))
        , outputIndex(outputIndx)
        , tolerance(tol)
    {
        qReg = reg;

        if (inputCount) {
            inputIndices = std::unique_ptr<bitLenInt>(new bitLenInt[inputCount]);
            std::copy(inputIndcs, inputIndcs + inputCount, inputIndices.get());
        }

        angles = std::unique_ptr<real1>(new real1[inputPower]());
    }

    /** Create a new QNeuron which is an exact duplicate of another, including its learned state. */
    QNeuron(const QNeuron& toCopy)
        : QNeuron(
              toCopy.qReg, toCopy.inputIndices.get(), toCopy.inputCount, toCopy.outputIndex, (real1_f)toCopy.tolerance)
    {
        std::copy(toCopy.angles.get(), toCopy.angles.get() + toCopy.inputPower, angles.get());
    }

    QNeuron& operator=(const QNeuron& toCopy)
    {
        qReg = toCopy.qReg;
        if (toCopy.inputCount) {
            inputIndices = NULL;
            inputIndices = std::unique_ptr<bitLenInt>(new bitLenInt[inputCount]);
            std::copy(toCopy.inputIndices.get(), toCopy.inputIndices.get() + inputCount, inputIndices.get());
        }
        angles = NULL;
        angles = std::unique_ptr<real1>(new real1[inputPower]());
        std::copy(toCopy.angles.get(), toCopy.angles.get() + toCopy.inputPower, angles.get());
        outputIndex = toCopy.outputIndex;
        tolerance = toCopy.tolerance;

        return *this;
    }

    /** Set the angles of this QNeuron */
    void SetAngles(real1* nAngles) { std::copy(nAngles, nAngles + inputPower, angles.get()); }

    /** Get the angles of this QNeuron */
    void GetAngles(real1* oAngles) { std::copy(angles.get(), angles.get() + inputPower, oAngles); }

    bitLenInt GetInputCount() { return inputCount; }

    bitCapInt GetInputPower() { return inputPower; }

    /** Feed-forward from the inputs, loaded in "qReg", to a binary categorical distinction. "expected" flips the binary
     * categories, if false. "resetInit," if true, resets the result qubit to 0.5/0.5 |0>/|1> superposition before
     * proceeding to predict. */
    real1_f Predict(bool expected = true, bool resetInit = true)
    {
        if (resetInit) {
            qReg->SetBit(outputIndex, false);
            qReg->RY((real1_f)(PI_R1 / 2), outputIndex);
        }

        if (!inputCount) {
            // If there are no controls, this "neuron" is actually just a bias.
            qReg->RY((real1_f)(angles.get()[0U]), outputIndex);
        } else {
            // Otherwise, the action can always be represented as a uniformly controlled gate.
            qReg->UniformlyControlledRY(inputIndices.get(), inputCount, outputIndex, angles.get());
        }
        real1_f prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1_F - prob;
        }
        return prob;
    }

    /** "Uncompute" the Predict() method */
    real1_f Unpredict(bool expected = true)
    {
        if (!inputCount) {
            // If there are no controls, this "neuron" is actually just a bias.
            qReg->RY((real1_f)(-angles.get()[0U]), outputIndex);
        } else {
            // Otherwise, the action can always be represented as a uniformly controlled gate.
            std::unique_ptr<real1> reverseAngles = std::unique_ptr<real1>(new real1[inputPower]);
            std::transform(angles.get(), angles.get() + inputPower, reverseAngles.get(), [](real1 r) { return -r; });
            qReg->UniformlyControlledRY(inputIndices.get(), inputCount, outputIndex, reverseAngles.get());
        }
        real1_f prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1_F - prob;
        }
        return prob;
    }

    real1_f LearnCycle(bool expected = true)
    {
        real1_f result = Predict(expected, false);
        Unpredict(expected);
        return result;
    }

    /** Perform one learning iteration, training all parameters
     *
     * Inputs must be already loaded into "qReg" before calling this method. "expected" is the true binary output
     * category, for training. "eta" is a volatility or "learning rate" parameter with a maximum value of 1.
     *
     * In the feedback process of learning, default initial conditions forward untrained predictions to 1/sqrt(2) * (|0>
     * + |1>) for the output bit. If you want to initialize other conditions before "Learn()," set "resetInit" to false.
     */
    void Learn(bool expected, real1_f eta, bool resetInit = true)
    {
        real1_f startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        for (bitCapInt perm = 0U; perm < inputPower; ++perm) {
            startProb = LearnInternal(expected, eta, perm, startProb);
            if (0 > startProb) {
                break;
            }
        }
    }

    /** Perform one learning iteration, measuring the entire QInterface and training the resulting permutation
     *
     * Inputs must be already loaded into "qReg" before calling this method. "expected" is the true binary output
     * category, for training. "eta" is a volatility or "learning rate" parameter with a maximum value of 1.
     *
     * In the feedback process of learning, default initial conditions forward untrained predictions to 1/sqrt(2) * (|0>
     * + |1>) for the output bit. If you want to initialize other conditions before "LearnPermutation()," set
     * "resetInit" to false.
     */
    void LearnPermutation(bool expected, real1_f eta, bool resetInit = true)
    {
        real1_f startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        bitCapInt perm = 0U;
        for (bitLenInt i = 0U; i < inputCount; ++i) {
            perm |= qReg->M(inputIndices.get()[i]) ? pow2(i) : 0U;
        }

        LearnInternal(expected, eta, perm, startProb);

        for (bitLenInt i = 0U; i < inputCount; ++i) {
            qReg->TrySeparate(inputIndices.get()[i]);
        }
    }

protected:
    real1_f LearnInternal(bool expected, real1_f eta, bitCapInt perm, real1_f startProb)
    {
        bitCapIntOcl permOcl = (bitCapIntOcl)perm;
        real1 endProb;
        real1 origAngle;

        origAngle = angles.get()[permOcl];

        // Try positive angle increment:
        angles.get()[permOcl] += eta * PI_R1;
        endProb = LearnCycle(expected);
        if ((ONE_R1 - endProb) <= tolerance) {
            return -ONE_R1_F;
        }
        if (endProb > startProb) {
            return (real1_f)endProb;
        }

        // If positive angle increment is not an improvement,
        // try negative angle increment:
        angles.get()[permOcl] -= 2 * eta * PI_R1;
        endProb = LearnCycle(expected);
        if ((ONE_R1 - endProb) <= tolerance) {
            return -ONE_R1_F;
        }
        if (endProb > startProb) {
            return (real1_f)endProb;
        }

        // If neither increment is an improvement,
        // restore the original variational parameter.
        angles.get()[permOcl] = origAngle;

        return (real1_f)startProb;
    }
};
} // namespace Qrack
