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

#include <future>

#include "oclengine.hpp"
#include "qengine_opencl_multi.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

QEngineOCLMulti::QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, int deviceCount, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
{
    maxQPower = 1 << qubitCount;
    
    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetNodeCount();
    }
    
    bitLenInt devPow = log2(deviceCount);
    
    subQubitCount = qubitCount >> (devPow - 1);
    subMaxQPower = 1 << subQubitCount;
    subBufferSize = sizeof(complex) * subMaxQPower >> 1;
    bool foundInitState = false;
    bitCapInt subInitVal;
        
    for (int i = 0; i < deviceCount; i++) {
        if ((!foundInitState) && (subMaxQPower * i > initState)) {
            subInitVal = initState - (subMaxQPower * i);
            foundInitState = true;
        }
        substateEngines.push_back(std::make_shared<QEngineOCL>(QEngineOCL(subQubitCount, subInitVal, rgp, i)));
        subInitVal = 0;
    }
}
    
void QEngineOCLMulti::ShuffleBuffers(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer) {
    queue->enqueueCopyBuffer(buff1, tempBuffer, subBufferSize, 0, subBufferSize);
    queue->finish();
    
    queue->enqueueCopyBuffer(buff2, buff1, 0, subBufferSize, subBufferSize);
    queue->finish();
    
    queue->enqueueCopyBuffer(tempBuffer, buff2, 0, subBufferSize, subBufferSize);
    queue->finish();
}
    
template<typename F, typename ... Args> void QEngineOCLMulti::SingleBitGate(bitLenInt order, F fn, Args ... gfnArgs) {
    int i;
    std::vector<std::future<void>> futures(substateEngines.size());
    if (order == 0) {
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i] = std::async(std::launch::async, [&]() { ((substateEngines[i].get())->*fn)(gfnArgs ...); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
    } else {
        cl::Context context = *(clObj->GetContextPtr());
        cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize);
        
        cl::Buffer buff1 = substateEngines[0]->GetStateBuffer();
        cl::Buffer buff2 = substateEngines[1]->GetStateBuffer();
        
        CommandQueuePtr queue = substateEngines[0]->GetQueuePtr();
        
        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
        
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i] = std::async(std::launch::async, [&]() { ((substateEngines[i].get())->*fn)(gfnArgs ...); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
        
        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
    }
}
    
void QEngineOCLMulti::SetQuantumState(complex* inputState) {
    throw "Not implemented";
}

void QEngineOCLMulti::SetPermutation(bitCapInt perm) {
    throw "Not implemented";
}
    
bitLenInt QEngineOCLMulti::Cohere(QInterfacePtr toCopy) {
    throw "Not implemented";
}
    
std::map<QInterfacePtr, bitLenInt> QEngineOCLMulti::Cohere(std::vector<QInterfacePtr> toCopy) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::H(bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (GFn)(&QEngineOCL::H), subQubitCount);
    }
    else {
        SingleBitGate(0, (GFn)(&QEngineOCL::H), qubitIndex);
    }
}
    
bool QEngineOCLMulti::M(bitLenInt qubitIndex) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::X(bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (GFn)(&QEngineOCL::X), subQubitCount);
    }
    else {
        SingleBitGate(0, (GFn)(&QEngineOCL::X), qubitIndex);
    }
}
    
void QEngineOCLMulti::Y(bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (GFn)(&QEngineOCL::Y), subQubitCount);
    }
    else {
        SingleBitGate(0, (GFn)(&QEngineOCL::Y), qubitIndex);
    }
}
    
void QEngineOCLMulti::Z(bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (GFn)(&QEngineOCL::Z), subQubitCount);
    }
    else {
        SingleBitGate(0, (GFn)(&QEngineOCL::Z), qubitIndex);
    }
}
    
void QEngineOCLMulti::CY(bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::CZ(bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "Not implemented";
}
void QEngineOCLMulti::RT(real1 radians, bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (RGFn)(&QEngineOCL::RT), radians, subQubitCount);
    }
    else {
        SingleBitGate(0, (RGFn)(&QEngineOCL::RT), radians, qubitIndex);
    }
}
void QEngineOCLMulti::RX(real1 radians, bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (RGFn)(&QEngineOCL::RX), radians, subQubitCount);
    }
    else {
        SingleBitGate(0, (RGFn)(&QEngineOCL::RX), radians, qubitIndex);
    }
}
void QEngineOCLMulti::CRX(real1 radians, bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
void QEngineOCLMulti::RY(real1 radians, bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (RGFn)(&QEngineOCL::RY), radians, subQubitCount);
    }
    else {
        SingleBitGate(0, (RGFn)(&QEngineOCL::RY), radians, qubitIndex);
    }
}
void QEngineOCLMulti::CRY(real1 radians, bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
void QEngineOCLMulti::RZ(real1 radians, bitLenInt qubitIndex) {
    bitLenInt order = qubitIndex - subQubitCount;
    if (order > 0) {
        SingleBitGate(order, (RGFn)(&QEngineOCL::RZ), radians, subQubitCount);
    }
    else {
        SingleBitGate(0, (RGFn)(&QEngineOCL::RZ), radians, qubitIndex);
    }
}
void QEngineOCLMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
void QEngineOCLMulti::CRT(real1 radians, bitLenInt control, bitLenInt target) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCSC(
                       bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECSC(
                       bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::ZeroPhaseFlip(bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
void QEngineOCLMulti::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) {
    throw "Not implemented";
}
void QEngineOCLMulti::PhaseFlip() {
    throw "Not implemented";
}
void QEngineOCLMulti::SetReg(bitLenInt start, bitLenInt length, bitCapInt value) {
    throw "Not implemented";
}
bitCapInt QEngineOCLMulti::MReg(bitLenInt start, bitLenInt length) {
    throw "Not implemented";
}
    
bitCapInt QEngineOCLMulti::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, unsigned char* values) {
    throw "Not implemented";
}
    
bitCapInt QEngineOCLMulti::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    throw "Not implemented";
}
bitCapInt QEngineOCLMulti::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    throw "Not implemented";
}
    
void QEngineOCLMulti::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) {
    throw "Not implemented";
}
void QEngineOCLMulti::CopyState(QInterfacePtr orig) {
    throw "Not implemented";
}
real1 QEngineOCLMulti::Prob(bitLenInt qubitIndex) {
    throw "Not implemented";
}
real1 QEngineOCLMulti::ProbAll(bitCapInt fullRegister) {
    throw "Not implemented";
}

} // namespace Qrack
