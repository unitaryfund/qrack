//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"
#include <algorithm>

namespace Qrack {

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

/**
 * Abstract QEngine implementation, for all "Schroedinger method" engines
 */
class QEngine : public QInterface {

public:
    QEngine(bitLenInt n, std::shared_ptr<std::default_random_engine> rgp = nullptr, bool doNorm = true)
        : QInterface(n, rgp, doNorm){};

    /** Destructor of QInterface */
    virtual ~QEngine(){};

    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, real1 nrmlzr = 1.0);
    using QInterface::M;
    virtual bool M(bitLenInt qubit);
    virtual bitCapInt M(const bitLenInt* bits, const bitLenInt& length);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    virtual void ApplyM(bitCapInt regMask, bool result, complex nrm);
    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) = 0;

    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);

    /// Swap values of two bits in register
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);

    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation) = 0;
    virtual void ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation) = 0;
    virtual void ProbMaskAll(const bitCapInt& mask, real1* probsArray);

protected:
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm) = 0;
    virtual void ApplyControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
        const complex* mtrx, bool doCalcNorm);
    virtual void ApplyControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyAntiControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
        const complex* mtrx, bool doCalcNorm);
    virtual void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyDoublyControlled2x2(
        bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm);
    virtual void ApplyDoublyAntiControlled2x2(
        bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm);
};
} // namespace Qrack
