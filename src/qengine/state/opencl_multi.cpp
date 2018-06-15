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

QEngineOCLMulti::QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, int deviceCount)
    : QInterface(qBitCount)
{
    runningNorm = 1.0;
    maxQPower = 1 << qubitCount;
    
    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetNodeCount();
    }
    
    bitLenInt devPow = log2(deviceCount);
    
    if (devPow > 1) {
        // TODO: Logic only safe up to two devices.
        devPow = 1;
    }
    
    subQubitCount = qubitCount - devPow;
    subMaxQPower = 1 << subQubitCount;
    subBufferSize = sizeof(complex) * subMaxQPower >> 1;
    bool foundInitState = false;
    bool partialInit = true;
    bitCapInt subInitVal;
 
    for (int i = 0; i < deviceCount; i++) {
        if ((!foundInitState) && ((subMaxQPower * (i + 1)) > initState)) {
            subInitVal = initState - (subMaxQPower * i);
            foundInitState = true;
            partialInit = false;
        }
        substateEngines.push_back(std::make_shared<QEngineOCL>(subQubitCount, subInitVal, rgp, i, partialInit));
        substateEngines[i]->EnableNormalize(false);
        subInitVal = 0;
        partialInit = true;
    }
}
    
void QEngineOCLMulti::ShuffleBuffers(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer) {
    queue->enqueueCopyBuffer(buff1, tempBuffer, subBufferSize, 0, subBufferSize);
    queue->flush();
    queue->finish();
    
    queue->enqueueCopyBuffer(buff2, buff1, 0, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();
    
    queue->enqueueCopyBuffer(tempBuffer, buff2, 0, 0, subBufferSize);
    queue->flush();
    queue->finish();
}
    
void QEngineOCLMulti::SwapBuffersLow(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer) {
    queue->enqueueCopyBuffer(buff1, tempBuffer, subBufferSize, 0, subBufferSize);
    queue->flush();
    queue->finish();
        
    queue->enqueueCopyBuffer(buff2, buff1, subBufferSize, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();
        
    queue->enqueueCopyBuffer(tempBuffer, buff2, 0, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();
}
    
template<typename F, typename ... Args> void QEngineOCLMulti::SingleBitGate(bool doNormalize, bitLenInt bit, F fn, Args ... gfnArgs) {
    int i;
    if (runningNorm != 1.0) {
        for (i = 0; i < substateEngines.size(); i++) {
            substateEngines[i]->SetNorm(runningNorm);
            substateEngines[i]->EnableNormalize(true);
        }
    }
    // This logic is only correct for up to 2 devices
    // TODO: Generalize logic to all powers of 2 devices
    std::vector<std::future<void>> futures(substateEngines.size());
    if (bit < subQubitCount) {
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, fn, bit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., bit); });
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
            QEngineOCLPtr engine = substateEngines[i];
            bitLenInt sqc = subQubitCount - 1;
            futures[i] = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
        
        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
    }
    
    if (doNormalize) {
        runningNorm = 0.0;
        for (i = 0; i < substateEngines.size(); i++) {
            runningNorm += substateEngines[i]->GetNorm();
            substateEngines[i]->EnableNormalize(false);
        }
    }
}
    
template<typename CF, typename F, typename ... Args> void QEngineOCLMulti::ControlledGate(bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs) {
    // This logic is only correct for up to 2 devices
    // TODO: Generalize logic to all powers of 2 devices
    int i;
    std::vector<std::future<void>> futures(substateEngines.size());
    if ((controlBit < subQubitCount) && (targetBit < subQubitCount)) {
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, cfn, controlBit, targetBit, gfnArgs ...]() { ((engine.get())->*cfn)(gfnArgs ..., controlBit, targetBit); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
    } else {
        bitLenInt max = substateEngines.size();
        bitLenInt min = max / 2;
        
        if (targetBit < subQubitCount) {
            for (i = min; i < max; i++) {
                QEngineOCLPtr engine = substateEngines[i];
                futures[i] = std::async(std::launch::async, [engine, fn, targetBit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., targetBit); });
            }
            for (i = min; i < max; i++) {
                futures[i].get();
            }
        } else {
            cl::Context context = *(clObj->GetContextPtr());
            cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize);
        
            cl::Buffer buff1 = substateEngines[0]->GetStateBuffer();
            cl::Buffer buff2 = substateEngines[1]->GetStateBuffer();
        
            CommandQueuePtr queue = substateEngines[0]->GetQueuePtr();
            
            ShuffleBuffers(queue, buff1, buff2, tempBuffer);
            
            for (i = min; i < max; i++) {
                QEngineOCLPtr engine = substateEngines[i];
                futures[i] = std::async(std::launch::async, [engine, fn, targetBit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., targetBit - 1); });
            }
            for (i = min; i < max; i++) {
                futures[i].get();
            }
            
            ShuffleBuffers(queue, buff1, buff2, tempBuffer);
        }
    }
}
    
template<typename CCF, typename CF, typename ... Args> void QEngineOCLMulti::DoublyControlledGate(bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt targetBit, CCF cfn, CF fn, Args ... gfnArgs) {
    // This logic is only correct for up to 2 devices
    // TODO: Generalize logic to all powers of 2 devices
    int i;
    std::vector<std::future<void>> futures(substateEngines.size());
    if ((controlBit1 < subQubitCount) && (controlBit2 < subQubitCount) && (targetBit < subQubitCount)) {
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, cfn, controlBit1, controlBit2, targetBit, gfnArgs ...]() { ((engine.get())->*cfn)(gfnArgs ..., controlBit1, controlBit2, targetBit); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
    } else {
        bitLenInt max = substateEngines.size();
        bitLenInt min = max / 2;
        bitLenInt controlBit = (controlBit1 < controlBit2) ? controlBit1 : controlBit2;
        if (targetBit < subQubitCount) {
            for (i = min; i < max; i++) {
                QEngineOCLPtr engine = substateEngines[i];
                futures[i] = std::async(std::launch::async, [engine, fn, controlBit, targetBit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., controlBit, targetBit); });
            }
            for (i = min; i < max; i++) {
                futures[i].get();
            }
        } else {
            cl::Context context = *(clObj->GetContextPtr());
            cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize);
                
            cl::Buffer buff1 = substateEngines[0]->GetStateBuffer();
            cl::Buffer buff2 = substateEngines[1]->GetStateBuffer();
                
            CommandQueuePtr queue = substateEngines[0]->GetQueuePtr();
                
            ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                
            for (i = min; i < max; i++) {
                QEngineOCLPtr engine = substateEngines[i];
                futures[i] = std::async(std::launch::async, [engine, fn, controlBit, targetBit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., controlBit, targetBit - 1); });
            }
            for (i = min; i < max; i++) {
                futures[i].get();
            }
                
            ShuffleBuffers(queue, buff1, buff2, tempBuffer);
        }
    }
}
    
void QEngineOCLMulti::SetQuantumState(complex* inputState) {
    throw "SetQuantumState not implemented";
}

void QEngineOCLMulti::SetPermutation(bitCapInt perm) {
    std::future<void> ftr;
    int i;
    int j = 0;
    for (i = 0; i < maxQPower; i+=subMaxQPower) {\
        if ((perm >= i) && (perm < (i + subMaxQPower))) {
            QEngineOCLPtr engine = substateEngines[j];
            bitCapInt p = perm - i;
            ftr = std::async(std::launch::async, [engine, p]() { engine->SetPermutation(p);});
        }
        else {
            substateEngines[j]->SetNorm(0.0);
            cl::Buffer buffer = substateEngines[j]->GetStateBuffer();
            CommandQueuePtr queue = substateEngines[j]->GetQueuePtr();
            queue->enqueueFillBuffer(buffer, complex(0.0, 0.0), 0, subMaxQPower);
            queue->flush();
        }
        j++;
    }
    for (i = 0; i < substateEngines.size(); i++) {
        CommandQueuePtr queue = substateEngines[i]->GetQueuePtr();
        queue->finish();
    }
    ftr.get();
}
    
bitLenInt QEngineOCLMulti::Cohere(QInterfacePtr toCopy) {
    throw "Cohere not implemented";
}
    
std::map<QInterfacePtr, bitLenInt> QEngineOCLMulti::Cohere(std::vector<QInterfacePtr> toCopy) {
    throw "Cohere not implemented";
}
    
void QEngineOCLMulti::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest) {
    throw "Decohere not implemented";
}
    
void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length) {
    throw "Dispose not implemented";
}
    
void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
    DoublyControlledGate(control1, control2, target, (CCGFn)(&QEngineOCL::CCNOT), (CGFn)(&QEngineOCL::CNOT));
}
    
void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::H(bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (GFn)(&QEngineOCL::H));
}
    
bool QEngineOCLMulti::M(bitLenInt qubit) {
    // TODO: Generalize to more than two devices.
    
    //if (runningNorm != 1.0) {
    //    NormalizeState();
    //}
    
    int i;
    
    real1 prob = Rand();
    real1 oneChance = Prob(qubit);
    
    bool result = ((prob < oneChance) && (oneChance > 0.0));
    real1 nrmlzr = 1.0;
    if (result) {
        if (oneChance > 0.0) {
            nrmlzr = oneChance;
        }
    } else {
        if (oneChance < 1.0) {
            nrmlzr = 1.0 - oneChance;
        }
    }
    
    if (qubit < subQubitCount) {
        std::vector<std::future<void>> futures(substateEngines.size());
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubit, result, nrmlzr]() {
                engine->ForceM(qubit, result, true, nrmlzr);
            });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            futures[i].get();
        }
    }
    else {
        bitLenInt init, max;
        if (result) {
            init = 1;
            max = 2;
        }
        else {
            init = 0;
            max = 1;
        }
        for (i = init; i < max; i++) {
            cl::Buffer buffer = substateEngines[i]->GetStateBuffer();
            CommandQueuePtr queue = substateEngines[i]->GetQueuePtr();
            queue->enqueueFillBuffer(buffer, complex(0.0, 0.0), 0, subMaxQPower);
            queue->flush();
        }
        for (i = init; i < max; i++) {
            CommandQueuePtr queue = substateEngines[i]->GetQueuePtr();
            queue->finish();
        }
    }
    
    return result;
}
    
void QEngineOCLMulti::X(bitLenInt qubitIndex) {
    SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::Y(bitLenInt qubitIndex) {
    SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Y));
}
    
void QEngineOCLMulti::Z(bitLenInt qubitIndex) {
    SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Z));
}
    
void QEngineOCLMulti::CY(bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CGFn)(&QEngineOCL::CY), (GFn)(&QEngineOCL::Y));
}
    
void QEngineOCLMulti::CZ(bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CGFn)(&QEngineOCL::CZ), (GFn)(&QEngineOCL::Z));
}
    
void QEngineOCLMulti::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "AND not implemented";
}
void QEngineOCLMulti::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "OR not implemented";
}
void QEngineOCLMulti::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    throw "XOR not implemented";
}
void QEngineOCLMulti::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "CLAND not implemented";
}
void QEngineOCLMulti::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "CLOR not implemented";
}
void QEngineOCLMulti::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit) {
    throw "CLXOR not implemented";
}
void QEngineOCLMulti::RT(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RT), radians);
}
void QEngineOCLMulti::RX(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::CRX(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CRGFn)(&QEngineOCL::CRX), (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::RY(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::CRY(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CRGFn)(&QEngineOCL::CRY), (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::RZ(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CRGFn)(&QEngineOCL::CRZ), (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRT(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(control, target, (CRGFn)(&QEngineOCL::CRT), (RGFn)(&QEngineOCL::RT), radians);
}
    
void QEngineOCLMulti::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) {
    throw "ROL not implemented";
}
void QEngineOCLMulti::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) {
    throw "ROR not implemented";
}
    
void QEngineOCLMulti::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "INC not implemented";
}
void QEngineOCLMulti::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "INCC not implemented";
}
void QEngineOCLMulti::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    throw "INCS not implemented";
}
void QEngineOCLMulti::INCSC(
                       bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    throw "INCSC not implemented";
}
void QEngineOCLMulti::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "INCSC not implemented";
}
void QEngineOCLMulti::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "INCBCD not implemented";
}
void QEngineOCLMulti::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "INCBCDC not implemented";
}
void QEngineOCLMulti::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
    throw "DEC not implemented";
}
void QEngineOCLMulti::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "DECC not implemented";
}
void QEngineOCLMulti::DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    throw "DECS not implemented";
}
void QEngineOCLMulti::DECSC(
                       bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    throw "DECSC not implemented";
}
void QEngineOCLMulti::DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "DECSC not implemented";
}
void QEngineOCLMulti::DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    throw "DECBCD not implemented";
}
void QEngineOCLMulti::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    throw "DECBCDC not implemented";
}
    
void QEngineOCLMulti::ZeroPhaseFlip(bitLenInt start, bitLenInt length) {
    throw "ZeroPhaseFlip not implemented";
}
void QEngineOCLMulti::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) {
    throw "CPhaseFlipIfLess not implemented";
}
void QEngineOCLMulti::PhaseFlip() {
    throw "PhaseFlip not implemented";
}
bitCapInt QEngineOCLMulti::MReg(bitLenInt start, bitLenInt length) {
    throw "MReg not implemented";
}
    
bitCapInt QEngineOCLMulti::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, unsigned char* values) {
    throw "IndexedLDA not implemented";
}
    
bitCapInt QEngineOCLMulti::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    throw "IndexedADC not implemented";
}
bitCapInt QEngineOCLMulti::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    throw "IndexedSBC not implemented";
}
    
void QEngineOCLMulti::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) {
    
    if (qubitIndex1 == qubitIndex2) {
        return;
    }
    
    int i;
    bitLenInt highBit, lowBit;
    if (qubitIndex1 > qubitIndex2) {
        highBit = qubitIndex1;
        lowBit = qubitIndex2;
    } else {
        lowBit = qubitIndex1;
        highBit = qubitIndex2;
    }
    // This logic is only correct for up to 2 devices
    // TODO: Generalize logic to all powers of 2 devices
    std::vector<std::future<void>> futures(substateEngines.size());
    if (highBit < subQubitCount) {
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex1, qubitIndex2]() { engine->Swap(qubitIndex1, qubitIndex2); });
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
        
        SwapBuffersLow(queue, buff1, buff2, tempBuffer);
        
        bitLenInt max = substateEngines.size();
        bitLenInt min = max / 2;
        for (i = min; i < max; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, lowBit]() { engine->X(lowBit); });
        }
        for (i = min; i < max; i++) {
            futures[i].get();
        }
        
        SwapBuffersLow(queue, buff1, buff2, tempBuffer);
    }
}
void QEngineOCLMulti::CopyState(QInterfacePtr orig) {
    throw "CopyState not implemented";
}
real1 QEngineOCLMulti::Prob(bitLenInt qubitIndex) {
    real1 oneChance = 0.0;
    int i;
    std::vector<std::future<real1>> futures(substateEngines.size());
    
    // This logic only works for up to two devices.
    // TODO: Generalize to higher numbers of devices
    if (qubitIndex < subQubitCount) {
        for (i = 0; i < substateEngines.size(); i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
        }
        for (i = 0; i < substateEngines.size(); i++) {
            oneChance += futures[i].get();
        }
    } else {
        for (i = substateEngines.size() / 2; i < substateEngines.size(); i++) {
            oneChance += substateEngines[i]->GetNorm();
        }
    }
    
    return oneChance;
}
real1 QEngineOCLMulti::ProbAll(bitCapInt fullRegister) {
    bitLenInt subIndex = fullRegister / subMaxQPower;
    fullRegister -= subIndex * subMaxQPower;
    return substateEngines[subIndex]->ProbAll(fullRegister);
}
    
} // namespace Qrack
