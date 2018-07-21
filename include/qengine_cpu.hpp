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

#include <algorithm>
#include <memory>

#include "qinterface.hpp"

#include "common/parallel_for.hpp"

namespace Qrack {

class QEngineCPU;
typedef std::shared_ptr<QEngineCPU> QEngineCPUPtr;

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);
template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

/**
 * General purpose QEngineCPU implementation
 */
class QEngineCPU : public QInterface, public ParallelFor {
protected:
    complex* stateVec;

public:
    QEngineCPU(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool partialInit = false);
    QEngineCPU(QEngineCPUPtr toCopy);
    ~QEngineCPU() { delete[] stateVec; }

    virtual void EnableNormalize(bool doN) { doNormalize = doN; }

    virtual void SetQuantumState(complex* inputState);

    virtual bitLenInt Cohere(QInterfacePtr toCopy) { return Cohere(std::dynamic_pointer_cast<QEngineCPU>(toCopy)); }
    std::map<QInterfacePtr, bitLenInt> Cohere(std::vector<QInterfacePtr> toCopy);

    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest);

    virtual bitLenInt Cohere(QEngineCPUPtr toCopy);
    virtual void Dispose(bitLenInt start, bitLenInt length);

    /** @} */

    /**
     * \defgroup RegGates Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     *
     * @{
     */

    using QInterface::X;
    virtual void X(bitLenInt start, bitLenInt length);
    using QInterface::CNOT;
    virtual void CNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QInterface::AntiCNOT;
    virtual void AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QInterface::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    using QInterface::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlip();
    virtual void SetPermutation(bitCapInt perm);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    using QInterface::Swap;
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual complex* GetStateVector();
    virtual void CopyState(QInterfacePtr orig);
    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual real1 GetNorm(bool update = true)
    {
        if (update) {
            UpdateRunningNorm();
        }
        return runningNorm;
    }
    virtual void SetNorm(real1 n) { runningNorm = n; }
    virtual void NormalizeState(real1 nrm = -999.0);

    /** @} */

protected:
    virtual void ResetStateVec(complex* nStateVec);
    void DecohereDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr dest);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm);
    virtual void UpdateRunningNorm();
    virtual complex* AllocStateVec(bitCapInt elemCount);
    virtual void ApplyM(bitCapInt qPower, bool result, complex nrm);
};
} // namespace Qrack
