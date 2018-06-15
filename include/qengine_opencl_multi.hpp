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

#include "qengine_opencl.hpp"

namespace Qrack {
    
class QEngineOCLMulti;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCLMulti : public QInterface {
protected:
    bitLenInt subQubitCount;
    bitCapInt subMaxQPower;
    real1 runningNorm;
    size_t subBufferSize;
    OCLEngine* clObj;
    std::vector<QEngineOCLPtr> substateEngines;
    std::vector<std::vector<cl::Buffer>> substateBuffers;
    
    uint32_t randomSeed;
    std::shared_ptr<std::default_random_engine> rand_generator;
    std::uniform_real_distribution<real1> rand_distribution;

public:
    QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr, int deviceCount = -1);
    
    virtual void SetQuantumState(complex* inputState);
    virtual void SetPermutation(bitCapInt perm);

    virtual bitLenInt Cohere(QInterfacePtr toCopy);
    virtual std::map<QInterfacePtr, bitLenInt> Cohere(std::vector<QInterfacePtr> toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);

    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void CNOT(bitLenInt control, bitLenInt target);
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);

    virtual void H(bitLenInt qubitIndex);
    virtual bool M(bitLenInt qubitIndex);
    virtual void X(bitLenInt qubitIndex);
    virtual void Y(bitLenInt qubitIndex);
    virtual void Z(bitLenInt qubitIndex);
    virtual void CY(bitLenInt control, bitLenInt target);
    virtual void CZ(bitLenInt control, bitLenInt target);

    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void RT(real1 radians, bitLenInt qubitIndex);
    virtual void RX(real1 radians, bitLenInt qubitIndex);
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target);
    virtual void RY(real1 radians, bitLenInt qubitIndex);
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target);
    virtual void RZ(real1 radians, bitLenInt qubitIndex);
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target);
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target);

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
    
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlip();
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);
    
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, unsigned char* values);
    
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void CopyState(QInterfacePtr orig);
    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbAll(bitCapInt fullRegister);

protected:
    typedef void (QEngineOCL::*GFn)(bitLenInt);
    typedef void (QEngineOCL::*RGFn)(real1, bitLenInt);
    typedef void (QEngineOCL::*CGFn)(bitLenInt, bitLenInt);
    typedef void (QEngineOCL::*CRGFn)(real1, bitLenInt, bitLenInt);
    template<typename F, typename ... Args> void SingleBitGate(bool doNormalize, bitLenInt bit, F fn, Args ... gfnArgs);
    template<typename CF, typename F, typename ... Args> void ControlledGate(bool anti, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs);
    
private:
    void ShuffleBuffers(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer);
    
    inline bitCapInt log2(bitCapInt n) {
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
