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
    rand_generator = rgp;
    
    runningNorm = 1.0;
    maxQPower = 1 << qubitCount;
    
    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetNodeCount();
    }
    
    bitLenInt devPow = log2(deviceCount);
    maxDeviceOrder = devPow;
    
    // Maximum of 2^N devices for N qubits:
    if (qubitCount <= devPow) {
        devPow = qubitCount - 1;
    }
    
    deviceCount = 1 << devPow;
    subEngineCount = deviceCount;
    
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
    int i, j;
    if (runningNorm != 1.0) {
        for (i = 0; i < subEngineCount; i++) {
            substateEngines[i]->SetNorm(runningNorm);
            substateEngines[i]->EnableNormalize(true);
        }
    }
    // This logic is only correct for up to 2 devices
    // TODO: Generalize logic to all powers of 2 devices
    if (bit < subQubitCount) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, fn, bit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., bit); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        std::vector<std::future<void>> futures(subEngineCount / 2);
        
        bitLenInt order = qubitCount - (bit + 1);
        bitLenInt groups = 1 << order;
        bitLenInt offset = 1 << ((qubitCount - subQubitCount) - (order + 1));
        
        cl::Context context = *(clObj->GetContextPtr());
        
        bitLenInt sqc = subQubitCount - 1;
        
        for (i = 0; i < groups; i++) {
            for (j = 0; j < offset; j++) {
                futures[j + (i * offset)] = std::async(std::launch::async, [this, context, offset, i, j, fn, sqc, gfnArgs ...]() {
                    cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize);
                    cl::Buffer buff1 = substateEngines[j + (i * offset)]->GetStateBuffer();
                    cl::Buffer buff2 = substateEngines[j + ((i + 1) * offset)]->GetStateBuffer();
                    QEngineOCLPtr engine = substateEngines[j + (i * offset)];
                    CommandQueuePtr queue = engine->GetQueuePtr();
                
                    ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                
                    std::future<void> future1 = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
                    engine = substateEngines[j + ((i + 1) * offset)];
                    std::future<void> future2 = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
                
                    future1.get();
                    future2.get();
                
                    ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                });
            }
        }
        for (i = 0; i < subEngineCount / 2; i++) {
            futures[i].get();
        }
    }
    
    if (doNormalize) {
        runningNorm = 0.0;
        for (i = 0; i < subEngineCount; i++) {
            runningNorm += substateEngines[i]->GetNorm();
            substateEngines[i]->EnableNormalize(false);
        }
    }
}
    
template<typename CF, typename F, typename ... Args> void QEngineOCLMulti::ControlledGate(bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs) {
    int i;
    
    if ((controlBit < subQubitCount) && (targetBit < subQubitCount)) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, cfn, controlBit, targetBit, gfnArgs ...]() { ((engine.get())->*cfn)(gfnArgs ..., controlBit, targetBit); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        ControlledBody(0, controlBit, targetBit, cfn, fn, gfnArgs ...);
    }
}
    
template<typename CCF, typename CF, typename F, typename ... Args> void QEngineOCLMulti::DoublyControlledGate(bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt targetBit, CCF ccfn, CF cfn, F fn, Args ... gfnArgs) {
    int i;
    
    if ((controlBit1 < subQubitCount) && (controlBit2 < subQubitCount) && (targetBit < subQubitCount)) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, ccfn, controlBit1, controlBit2, targetBit, gfnArgs ...]() { ((engine.get())->*ccfn)(gfnArgs ..., controlBit1, controlBit2, targetBit); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else  {
        bitLenInt lowControl, highControl;
        if (controlBit1 < controlBit2) {
            lowControl = controlBit1;
            highControl = controlBit2;
        }
        else {
            lowControl = controlBit2;
            highControl = controlBit1;
        }
        
        if (lowControl < subQubitCount) {
            // CNOT logic
            ControlledBody(0, highControl, targetBit, ccfn, cfn, gfnArgs ..., lowControl);
        }
        else {
            // Skip first group, if more than one group.
            ControlledBody(1, highControl, targetBit, cfn, fn, gfnArgs ...);
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
    for (i = 0; i < subEngineCount; i++) {
        CommandQueuePtr queue = substateEngines[i]->GetQueuePtr();
        queue->finish();
    }
    ftr.get();
}
    
bitLenInt QEngineOCLMulti::Cohere(QEngineOCLMultiPtr toCopy) {
    int i, j;
    
    bitLenInt result = qubitCount;
    
    cl::Context context = *(clObj->GetContextPtr());
    CommandQueuePtr queue;
    QEngineOCLPtr toAddEngine;
    
    size_t divSize = subBufferSize << 1;
    size_t divCount = 1 << (toCopy->subQubitCount);
    bitLenInt copySubEngineCount = toCopy->subEngineCount;
    
    std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount * copySubEngineCount);
    
    std::vector<std::future<void>> cohereFutures(copySubEngineCount);
    
    for (i = 0; i < copySubEngineCount; i++) {
        // Putting these two statements inside of the lambda led to a SIGILL on the test machine, probably due to a race condition.
        toAddEngine = toCopy->substateEngines[i];
        queue = toAddEngine->GetQueuePtr();
        
        cohereFutures[i] = std::async(std::launch::async, [this, i, divSize, divCount, toCopy, toAddEngine, queue, &nSubstateEngines]() {
            int j, destIndex, destEngine, sourceIndex, sourceEngine;
            
            std::vector<QEngineOCLPtr> tempEngines(subEngineCount);
            
            for (j = 0; j < subEngineCount; j++) {
                nSubstateEngines[j + (i * subEngineCount)] = std::make_shared<QEngineOCL>(substateEngines[j]);
                nSubstateEngines[j + (i * subEngineCount)]->Cohere(toAddEngine);
                tempEngines[j] = std::make_shared<QEngineOCL>(nSubstateEngines[j + (i * subEngineCount)]);
            }
            
            if (subEngineCount == 1) {
                return;
            }
        
            for (j = 0; j < (subEngineCount * divCount); j++) {
                destEngine = (j / divCount) + (i * subEngineCount);
                destIndex = j & (divCount - 1) ;
                sourceEngine = j & (subEngineCount - 1);
                sourceIndex = j / subEngineCount;
            
                queue->enqueueCopyBuffer(
                    tempEngines[sourceEngine]->GetStateBuffer(),
                    nSubstateEngines[destEngine]->GetStateBuffer(),
                    sourceIndex * divSize, destIndex * divSize, divSize);
                queue->flush();
                queue->finish();
            }
        });
    }
    
    for (i = 0; i < copySubEngineCount; i++) {
        cohereFutures[i].get();
    }
    
    size_t sbSize = (subBufferSize << (toCopy->subQubitCount)) << 1;
    if (subEngineCount != copySubEngineCount) {
        substateEngines.resize(copySubEngineCount);
    }
    SetQubitCount(qubitCount + (toCopy->GetQubitCount()));
    
    for (i = 0; i < copySubEngineCount; i++) {
        QEngineOCLPtr toReplace = std::make_shared<QEngineOCL>(subQubitCount, 0, rand_generator, i, true);
        toReplace->EnableNormalize(false);
        for (j = 0; j < subEngineCount; j++) {
            queue->enqueueCopyBuffer(
                nSubstateEngines[j + (i * subEngineCount)]->GetStateBuffer(),
                toReplace->GetStateBuffer(),
                0, j * sbSize, sbSize);
            queue->flush();
            queue->finish();
        }
        substateEngines[i] = toReplace;
    }
    subEngineCount = copySubEngineCount;
    
    // If we have more capacity for nodes, due to extra qubits, split it up now.
    bitLenInt diffOrder = (1 << maxDeviceOrder) - subEngineCount;
    if (diffOrder > (toCopy->GetQubitCount())) {
        diffOrder = toCopy->GetQubitCount();
    }
    if (diffOrder > 0) {
        std::vector<std::future<void>> futures(subEngineCount);
        bitLenInt diffPower = 1 << (diffOrder - 1);
        std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount * diffOrder);
        for (i = 0; i < subEngineCount; i++) {
            toAddEngine = substateEngines[i];
            queue = toAddEngine->GetQueuePtr();
            futures[i] = std::async(std::launch::async, [this, i, diffOrder, diffPower, toAddEngine, queue, &nSubstateEngines]() {
                for (int j = 0; j < diffOrder; j++) {
                    nSubstateEngines.push_back(std::make_shared<QEngineOCL>(subQubitCount - diffOrder, 0, rand_generator, i, true));
                    nSubstateEngines[i]->EnableNormalize(false);
                    queue->enqueueCopyBuffer(
                        toAddEngine->GetStateBuffer(),
                        nSubstateEngines[j + (i * diffOrder)]->GetStateBuffer(),
                        (subBufferSize * j) / diffPower, 0, subBufferSize / diffPower);
                    queue->flush();
                    queue->finish();
                }
            });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
        substateEngines = nSubstateEngines;
        
        subEngineCount = substateEngines.size();
        SetQubitCount(qubitCount);
    }
    
    return result;
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
    DoublyControlledGate(control1, control2, target, (CCGFn)(&QEngineOCL::CCNOT), (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
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
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubit, result, nrmlzr]() {
                engine->ForceM(qubit, result, true, nrmlzr);
            });
        }
        for (i = 0; i < subEngineCount; i++) {
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

    if ((qubitIndex1 < subQubitCount) && (qubitIndex2 < subQubitCount)) {
        // Here, it's entirely contained within single nodes:
        std::vector<std::future<void>> futures(subEngineCount);
        int i;
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex1, qubitIndex2]() { engine->Swap(qubitIndex1, qubitIndex2); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else if ((qubitIndex1 >= subQubitCount) && (qubitIndex2 >= subQubitCount)) {
        // Here, it's possible to swap entire engines:
        qubitIndex1 -= subQubitCount;
        qubitIndex2 -= subQubitCount;
        
        bitCapInt bit1Mask = 1 << qubitIndex1;
        bitCapInt bit2Mask = 1 << qubitIndex2;
        bitCapInt otherMask = subEngineCount - 1;
        otherMask ^= bit1Mask | bit2Mask;
        
        std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount);
        
        par_for(0, 1 << (qubitCount - subQubitCount), [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = (lcv & otherMask);
            bitCapInt bit1Res = ((lcv & bit1Mask) >> qubitIndex1) << qubitIndex2;
            bitCapInt bit2Res = ((lcv & bit2Mask) >> qubitIndex2) << qubitIndex1;
            nSubstateEngines[bit1Res | bit2Res | otherRes] = substateEngines[lcv];
        });
        
        substateEngines = nSubstateEngines;
    } else {
        // "Swap" is tricky, if we're distributed across nodes.
        // However, we get it virtually for free in a QUnit, so this is a low-priority case.
        // Assuming our CNOT works, so does this:
        CNOT(qubitIndex1, qubitIndex2);
        CNOT(qubitIndex2, qubitIndex1);
        CNOT(qubitIndex1, qubitIndex2);
    }
}
void QEngineOCLMulti::CopyState(QInterfacePtr orig) {
    throw "CopyState not implemented";
}
real1 QEngineOCLMulti::Prob(bitLenInt qubitIndex) {
    real1 oneChance = 0.0;
    int i;
    std::vector<std::future<real1>> futures(subEngineCount);
    
    // This logic only works for up to two devices.
    // TODO: Generalize to higher numbers of devices
    if (qubitIndex < subQubitCount) {
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
        }
        for (i = 0; i < subEngineCount; i++) {
            oneChance += futures[i].get();
        }
    } else {
        for (i = subEngineCount / 2; i < subEngineCount; i++) {
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
    
template<typename CF, typename F, typename ... Args> void QEngineOCLMulti::ControlledBody(bitLenInt controlDepth, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs) {
    int i, j, k;
    
    std::vector<std::future<void>> futures(subEngineCount / 2);
    cl::Context context = *(clObj->GetContextPtr());
    
    bitLenInt order, groups, diffOrder, pairOffset, groupOffset;
    bool useTopDevice;
    
    if (targetBit < controlBit) {
        int pairPower =subEngineCount - (1 << (qubitCount - (targetBit + 1)));
        
        if (pairPower <= 0) {
            
            // Simple special case:
            bitLenInt offset = 1 << (qubitCount - (controlBit + 1));
            for (i = 0; i < (subEngineCount / 2); i++) {
                QEngineOCLPtr engine = substateEngines[(i * offset) + (subEngineCount / (2 *  offset))];
                futures[i] = std::async(std::launch::async, [engine,fn, targetBit, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., targetBit); });
            }
            for (i = 0; i < subEngineCount / 2; i++) {
                futures[i].get();
            }
            
            // We're done with the controlled gate:
            return;
        }
        else {
            pairOffset = log2(pairPower);
        }
        
        order = qubitCount - (controlBit + 1);
        diffOrder = (targetBit < subQubitCount) ? (controlBit - subQubitCount) : ((controlBit - targetBit) - 1);
        useTopDevice = true;
    } else {
        order = qubitCount - (targetBit + 1);
        diffOrder = (controlBit < subQubitCount) ? (targetBit - subQubitCount) : ((targetBit - controlBit) - 1);
        pairOffset = (subEngineCount / (2 * (1 << order))) << diffOrder;
        useTopDevice = subEngineCount > 2;
    }
    groups = 1 << order;
    groupOffset = subEngineCount / (2 * groups) >> (diffOrder - order);
    
    bitLenInt sqc = subQubitCount - 1;
    
    bitLenInt index;
    
    bitLenInt controlPower = 1 << controlDepth;
    bitLenInt firstGroup = groups - (groups / controlPower);
    if (firstGroup >= groups) {
        firstGroup = 0;
    }
    
    for (i = firstGroup; i < groups; i++) {
        for (j = 0; j < pairOffset; j++) {
            for (k = 0; k < groupOffset; k++) {
                index = groupOffset / 2 + j + (i * groupOffset);
                futures[k + (j * groupOffset) + (i * groupOffset * pairOffset)] = std::async(std::launch::async, [this, useTopDevice, context, pairOffset, index, fn, sqc, gfnArgs ...]() {
                    cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize);
                    cl::Buffer buff1 = substateEngines[index]->GetStateBuffer();
                    cl::Buffer buff2 = substateEngines[index + pairOffset]->GetStateBuffer();
                    QEngineOCLPtr engine = substateEngines[index];
                    CommandQueuePtr queue = engine->GetQueuePtr();
                    
                    ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                    
                    if (useTopDevice) {
                        std::future<void> future1 = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
                        engine = substateEngines[index + pairOffset];
                        std::future<void> future2 = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
                        
                        future1.get();
                        future2.get();
                    }
                    else {
                        engine = substateEngines[index + pairOffset];
                        std::future<void> future = std::async(std::launch::async, [engine, fn, sqc, gfnArgs ...]() { ((engine.get())->*fn)(gfnArgs ..., sqc); });
                        
                        future.get();
                    }
                    
                    ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                });
            }
        }
    }
    for (i = 0; i < subEngineCount / 2; i++) {
        futures[i].get();
    }
}
    
} // namespace Qrack
