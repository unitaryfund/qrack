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

#include "common/oclengine.hpp"
#include "common/parallel_for.hpp"
#include "qengine_opencl.hpp"

namespace Qrack {

class QEngineOCLMulti;
typedef std::shared_ptr<QEngineOCLMulti> QEngineOCLMultiPtr;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCLMulti : public QInterface, public ParallelFor {
protected:
    bitLenInt subQubitCount;
    bitCapInt subMaxQPower;
    bitLenInt subEngineCount;
    bitLenInt maxDeviceOrder;
    OCLEngine* clObj;
    std::vector<QEngineOCLPtr> substateEngines;
    std::vector<std::vector<cl::Buffer>> substateBuffers;
    std::vector<int> deviceIDs;

public:
    /**
     * Initialize a Qrack::QEngineOCLMulti object. Specify the number of qubits and an initial permutation state.
     * Additionally, optionally specify a pointer to a random generator engine object and a number of sub-engines,
     * (usually one per device, though this can be over-allocated,) to break the object into. The "deviceCount" should
     * be a power of 2, but it will be floored to a power of two if the parameter is not already a power of two. The
     * QEngineOCL can not use more than 1 power of 2 devices per qubit. (2^N devices for N qubits.) Powers of 2 in
     * excess of the qubit count will only be used if this engine acquires additional qubits.
     */
    QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr,
        int deviceCount = -1);

    /**
     * Initialize a Qrack::QEngineOCLMulti object. Specify the number of qubits and an initial permutation state.
     * Additionally, optionally specify a list of device IDs for sub-engines and a pointer to a random generator engine
     * object.
     *
     * "devIDs" is a list of integers that represent the index of OpenCL devices in the OCLEngine singleton, to select
     * how equal sized sub-engines are distributed between devices in this engine. The QEngineOCLMulti will only have a
     * power of 2 count of subengines at a time, and not more than 1 power of 2 devices per qubit. (2^N devices for N
     * qubits.) Devices in excess of the highest power of two in the list count will essentially be ignored. Powers of 2
     * in excess of the qubit count will only be used if this engine acquires additional qubits. It might be possible to
     * load balance this way, for example, by allocating 3 sub-engines on one device index and one sub-engine on a
     * second device index. (Whether this is an efficient load balancing mechanism will depend on the particulars of the
     * system architecture and instance initialization.)
     */
    QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, std::vector<int> devIDs,
        std::shared_ptr<std::default_random_engine> rgp = nullptr);

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
        subEngineCount = substateEngines.size();
        subQubitCount = qubitCount - log2(subEngineCount);
        subMaxQPower = 1 << subQubitCount;
    }

    virtual void SetQuantumState(complex* inputState);
    virtual void SetPermutation(bitCapInt perm);

    virtual bitLenInt Cohere(QEngineOCLMultiPtr toCopy);
    virtual bitLenInt Cohere(QInterfacePtr toCopy)
    {
        return Cohere(std::dynamic_pointer_cast<QEngineOCLMulti>(toCopy));
    }
    virtual std::map<QInterfacePtr, bitLenInt> Cohere(std::vector<QInterfacePtr> toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QEngineOCLMultiPtr dest);
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decohere(start, length, std::dynamic_pointer_cast<QEngineOCLMulti>(dest));
    }
    virtual void Dispose(bitLenInt start, bitLenInt length);

    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);

    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void CNOT(bitLenInt control, bitLenInt target);
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);

    virtual void H(bitLenInt qubitIndex);
    virtual bool M(bitLenInt qubitIndex);
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, real1 nrmlzr = 1.0);
    virtual void X(bitLenInt qubitIndex);
    virtual void Y(bitLenInt qubitIndex);
    virtual void Z(bitLenInt qubitIndex);
    virtual void CY(bitLenInt control, bitLenInt target);
    virtual void CZ(bitLenInt control, bitLenInt target);

    virtual void RT(real1 radians, bitLenInt qubitIndex);
    virtual void RX(real1 radians, bitLenInt qubitIndex);
    virtual void RY(real1 radians, bitLenInt qubitIndex);
    virtual void RZ(real1 radians, bitLenInt qubitIndex);
    virtual void Exp(real1 radians, bitLenInt qubitIndex);
    virtual void ExpX(real1 radians, bitLenInt qubitIndex);
    virtual void ExpY(real1 radians, bitLenInt qubitIndex);
    virtual void ExpZ(real1 radians, bitLenInt qubitIndex);
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target);
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target);
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target);
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target);

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
    virtual void MUL(
        bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bool clearCarry = false);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit,
        bitLenInt length, bool clearCarry = false);
    virtual void CDIV(
        bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit, bitLenInt length);

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);

    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);
    virtual void CopyState(QInterfacePtr orig) { CopyState(std::dynamic_pointer_cast<QEngineOCLMulti>(orig)); }
    virtual void CopyState(QEngineOCLMultiPtr orig);
    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual bool IsPhaseSeparable(bool forceCheck = false);

    virtual void X(bitLenInt start, bitLenInt length);
    virtual void CNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);

protected:
    typedef void (QEngineOCL::*GFn)(bitLenInt);
    typedef void (QEngineOCL::*RGFn)(real1, bitLenInt);
    typedef void (QEngineOCL::*CGFn)(bitLenInt, bitLenInt);
    typedef void (QEngineOCL::*CRGFn)(real1, bitLenInt, bitLenInt);
    typedef void (QEngineOCL::*CCGFn)(bitLenInt, bitLenInt, bitLenInt);
    typedef void (QEngineOCL::*ASBFn)(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);
    template <typename F, typename... Args> void SingleBitGate(bool doNormalize, bitLenInt bit, F fn, Args... gfnArgs);
    template <typename CF, typename F, typename... Args>
    void ControlledGate(bool anti, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args... gfnArgs);
    template <typename CCF, typename CF, typename F, typename... Args>
    void DoublyControlledGate(bool anti, bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt targetBit, CCF ccfn,
        CF cfn, F fn, Args... gfnArgs);

    template <typename F, typename OF> void RegOp(F fn, OF ofn, bitLenInt length, std::vector<bitLenInt> bits);

    // For scalable cluster distribution, these methods should ultimately be entirely removed:
    void CombineEngines(bitLenInt bit);
    void SeparateEngines();
    template <typename F> void CombineAndOp(F fn, std::vector<bitLenInt> bits);

    void NormalizeState(real1 nrm = -999.0);

    void MetaX(bitLenInt start, bitLenInt length);
    void MetaCNOT(bool anti, std::vector<bitLenInt> controls, bitLenInt target);
    template <typename F, typename... Args>
    void MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, F fn, Args... gfnArgs);
    template <typename F, typename... Args>
    void SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, F fn, Args... gfnArgs);
    template <typename F, typename... Args>
    void ControlledSkip(bool anti, bitLenInt controlDepth, bitLenInt targetBit, F fn, Args... gfnArgs);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm)
    {
        throw "Apply2x2 not implemented in interface";
    }
    void ApplyM(bitCapInt qPower, bool result, complex nrm) { throw "ApplyM not implemented in interface"; }

private:
    void Init(bitLenInt qBitCount, bitCapInt initState);

    void ShuffleBuffers(QEngineOCLPtr engine1, QEngineOCLPtr engine2);

    bitLenInt SeparateMetaCNOT(bool anti, std::vector<bitLenInt> controls, bitLenInt target, bitLenInt length);

    inline bitCapInt log2(bitCapInt n)
    {
        bitLenInt pow = 0;
        bitLenInt p = n >> 1;
        while (p != 0) {
            p >>= 1;
            pow++;
        }
        return pow;
    }
};
} // namespace Qrack
