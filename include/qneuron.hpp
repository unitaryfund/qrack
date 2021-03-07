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
    QInterfacePtr qReg;
    bitLenInt* inputIndices;
    bitLenInt inputCount;
    bitCapIntOcl inputPower;
    bitLenInt outputIndex;
    real1* angles;
    real1 tolerance;

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

        if (inputCount > 0) {
            inputIndices = new bitLenInt[inputCount];
            std::copy(inputIndcs, inputIndcs + inputCount, inputIndices);
        }

        angles = new real1[inputPower]();
    }

    /** Create a new QNeuron which is an exact duplicate of another, including its learned state. */
    QNeuron(const QNeuron& toCopy)
        : QNeuron(toCopy.qReg, toCopy.inputIndices, toCopy.inputCount, toCopy.outputIndex, toCopy.tolerance)
    {
        std::copy(toCopy.angles, toCopy.angles + toCopy.inputPower, angles);
    }

    ~QNeuron()
    {
        if (inputCount > 0) {
            delete[] inputIndices;
        }
        delete[] angles;
    }

    /** Set the angles of this QNeuron */
    void SetAngles(real1* nAngles) { std::copy(nAngles, nAngles + inputPower, angles); }

    /** Get the angles of this QNeuron */
    void GetAngles(real1* oAngles) { std::copy(angles, angles + inputPower, oAngles); }

    bitLenInt GetInputCount() { return inputCount; }

    bitCapInt GetInputPower() { return inputPower; }

    /** Feed-forward from the inputs, loaded in "qReg", to a binary categorical distinction. "expected" flips the binary
     * categories, if false. "resetInit," if true, resets the result qubit to 0.5/0.5 |0>/|1> superposition before
     * proceeding to predict. */
    real1_f Predict(bool expected = true, bool resetInit = true)
    {
        if (resetInit) {
            qReg->SetBit(outputIndex, false);
            qReg->RY(M_PI_2, outputIndex);
        }

        if (inputCount == 0) {
            // If there are no controls, this "neuron" is actually just a bias.
            qReg->RY(angles[0], outputIndex);
        } else {
            // Otherwise, the action can always be represented as a uniformly controlled gate.
            qReg->UniformlyControlledRY(inputIndices, inputCount, outputIndex, angles);
        }
        real1 prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1 - prob;
        }
        return prob;
    }

    /** "Uncompute" the Predict() method */
    real1_f Unpredict(bool expected = true)
    {
        if (inputCount == 0) {
            // If there are no controls, this "neuron" is actually just a bias.
            qReg->RY(-angles[0], outputIndex);
        } else {
            // Otherwise, the action can always be represented as a uniformly controlled gate.
            real1* reverseAngles = new real1[inputPower];
            std::transform(angles, angles + inputPower, reverseAngles, [](real1_f r) { return -r; });
            qReg->UniformlyControlledRY(inputIndices, inputCount, outputIndex, reverseAngles);
            delete[] reverseAngles;
        }
        real1 prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1 - prob;
        }
        return prob;
    }

    real1_f LearnCycle(bool expected = true)
    {
        real1 result = Predict(expected, false);
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
        real1 startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        for (bitCapInt perm = 0; perm < inputPower; perm++) {
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
        real1 startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        bitCapInt perm = 0;
        for (bitLenInt i = 0; i < inputCount; i++) {
            perm |= qReg->M(inputIndices[i]) ? pow2(i) : 0;
        }

        LearnInternal(expected, eta, perm, startProb);

        for (bitLenInt i = 0; i < inputCount; i++) {
            qReg->TrySeparate(inputIndices[i]);
        }
    }

protected:
    real1_f LearnInternal(bool expected, real1_f eta, bitCapInt perm, real1_f startProb)
    {
        bitCapIntOcl permOcl = (bitCapIntOcl)perm;
        real1 endProb;
        real1 origAngle;

        origAngle = angles[permOcl];

        // Try positive angle increment:
        angles[permOcl] += eta * M_PI;
        endProb = LearnCycle(expected);
        if ((ONE_R1 - endProb) <= tolerance) {
            return -ONE_R1;
        }
        if (endProb > startProb) {
            return endProb;
        }

        // If positive angle increment is not an improvement,
        // try negative angle increment:
        angles[permOcl] -= 2 * eta * M_PI;
        endProb = LearnCycle(expected);
        if ((ONE_R1 - endProb) <= tolerance) {
            return -ONE_R1;
        }
        if (endProb > startProb) {
            return endProb;
        }

        // If neither increment is an improvement,
        // restore the original variational parameter.
        angles[permOcl] = origAngle;

        return startProb;
    }
};
} // namespace Qrack
