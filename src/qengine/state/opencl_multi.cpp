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
#include <iostream>

#include "oclengine.hpp"
#include "qengine_opencl_multi.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

QEngineOCLMulti::QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, int deviceCount)
    : QInterface(qBitCount)
{
    rand_generator = rgp;
    
    runningNorm = 1.0;
    
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
    
    if (deviceCount == 1) {
        substateEngines.push_back(std::make_shared<QEngineOCL>(qubitCount, initState, rgp));
        return;
    }
 
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
    
    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*fn)(gfnArgs ..., bit);
        return;
    }
    
    int i, j;
    if (runningNorm != 1.0) {
        NormalizeState();
    }
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
        
        if ((offset == 0) || (offset > 1)) {
            std::cout<<"offset="<<(int)offset<<std::endl;
        }
        
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
        }
    }
}
    
template<typename CF, typename F, typename ... Args> void QEngineOCLMulti::ControlledGate(bool anti, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs) {
        
    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*cfn)(gfnArgs ..., controlBit, targetBit);
        return;
    }
        
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
        CombineAndOp([&](QEngineOCLPtr engine) {
            (engine.get()->*cfn)(gfnArgs ..., controlBit, targetBit);
        }, {controlBit, targetBit});
        //ControlledBody(anti, 0, controlBit, targetBit, cfn, fn, gfnArgs ...);
    }
}
    
template<typename CCF, typename CF, typename F, typename ... Args> void QEngineOCLMulti::DoublyControlledGate(bool anti, bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt targetBit, CCF ccfn, CF cfn, F fn, Args ... gfnArgs) {
        
    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*ccfn)(gfnArgs ..., controlBit1, controlBit2, targetBit);
        return;
    }
        
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
            
        CombineAndOp([&](QEngineOCLPtr engine) {
            (engine.get()->*ccfn)(gfnArgs ..., controlBit1, controlBit2, targetBit);
        }, {controlBit1, controlBit2, targetBit});
#if 0
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
            ControlledBody(anti, 0, highControl, targetBit, ccfn, cfn, gfnArgs ..., lowControl);
        }
        else {
            // Skip first group, if more than one group.
            ControlledBody(anti, 1, highControl, targetBit, cfn, fn, gfnArgs ...);
        }
#endif
    }
}

    
void QEngineOCLMulti::SetQuantumState(complex* inputState) {
    CombineAllEngines();
    substateEngines[0]->SetQuantumState(inputState);
    SeparateAllEngines();
}

void QEngineOCLMulti::SetPermutation(bitCapInt perm) {
    if (subEngineCount == 1) {
        substateEngines[0]->SetPermutation(perm);
        return;
    }
    
    std::future<void> ftr;
    int i;
    int j = 0;
    for (i = 0; i < maxQPower; i+=subMaxQPower) {
        if ((perm >= i) && (perm < (i + subMaxQPower))) {
            QEngineOCLPtr engine = substateEngines[j];
            bitCapInt p = perm - i;
            ftr = std::async(std::launch::async, [engine, p]() { engine->SetPermutation(p);});
        }
        else {
            cl::Buffer buffer = substateEngines[j]->GetStateBuffer();
            CommandQueuePtr queue = substateEngines[j]->GetQueuePtr();
            queue->enqueueFillBuffer(buffer, complex(0.0, 0.0), 0, subBufferSize << 1);
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
 
#if 0
bitLenInt QEngineOCLMulti::Cohere(QEngineOCLMultiPtr toCopy) {
    int i, j, destIndex, destEngine, sourceIndex, sourceEngine;
    QEngineOCLPtr toAddEngine, origEngine;
    
    bitLenInt result = qubitCount;
    
    cl::Context context = *(clObj->GetContextPtr());
    CommandQueuePtr queue;
    
    size_t divSize = subBufferSize << 1;
    size_t divCount = 1 << (toCopy->subQubitCount);
    bitLenInt copySubEngineCount = toCopy->subEngineCount;
    
    std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount * copySubEngineCount);
    
    for (i = 0; i < copySubEngineCount; i++) {
        toAddEngine = toCopy->substateEngines[i];
        queue = toAddEngine->GetQueuePtr();
        std::vector<QEngineOCLPtr> tempEngines(subEngineCount);
        for (j = 0; j < subEngineCount; j++) {
            nSubstateEngines[j + (i * subEngineCount)] = std::make_shared<QEngineOCL>(substateEngines[j]);
            nSubstateEngines[j + (i * subEngineCount)]->Cohere(toAddEngine);
            tempEngines[j] = std::make_shared<QEngineOCL>(nSubstateEngines[j + (i * subEngineCount)]);
        }
        
        if (subEngineCount == 1) {
            break;
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
    SetQubitCount(qubitCount);
    
    // If we have more capacity for nodes, due to extra qubits, split it up now.
    int diffOrder = (1 << maxDeviceOrder) - subEngineCount;
    if (diffOrder > (toCopy->GetQubitCount())) {
        diffOrder = toCopy->GetQubitCount();
    }
    
    if (diffOrder > 0) {
        bitLenInt diffPower = 1 << diffOrder;
        std::vector<QEngineOCLPtr> nSubstateEngines;
        for (i = 0; i < subEngineCount; i++) {
            toAddEngine = substateEngines[i];
            queue = toAddEngine->GetQueuePtr();
            for (int j = 0; j < diffOrder; j++) {
                nSubstateEngines.push_back(std::make_shared<QEngineOCL>(subQubitCount - diffOrder, 0, rand_generator, i, true));
                nSubstateEngines[j + (i * diffOrder)]->EnableNormalize(false);
                queue->enqueueCopyBuffer(
                    toAddEngine->GetStateBuffer(),
                    nSubstateEngines[j + (i * diffOrder)]->GetStateBuffer(),
                    ((subBufferSize << 1) * j) / diffPower, 0, (subBufferSize << 1) / diffPower);
                queue->flush();
                queue->finish();
            }
        }
        
        substateEngines = nSubstateEngines;
        SetQubitCount(qubitCount);
    }

    return result;
}

std::map<QInterfacePtr, bitLenInt> QEngineOCLMulti::Cohere(std::vector<QInterfacePtr> toCopy) {
    std::map<QInterfacePtr, bitLenInt> ret;
    
    for (auto&& q : toCopy) {
        ret[q] = Cohere(q);
    }
    
    return ret;
}
#endif
    
bitLenInt QEngineOCLMulti::Cohere(QEngineOCLMultiPtr toCopy) {
    bitLenInt result;
    CombineAllEngines();
    toCopy->CombineAllEngines();
    result = substateEngines[0]->Cohere(toCopy->substateEngines[0]);
    SetQubitCount(qubitCount + toCopy->qubitCount);
    SeparateAllEngines();
    toCopy->SeparateAllEngines();
    return result;
}

std::map<QInterfacePtr, bitLenInt> QEngineOCLMulti::Cohere(std::vector<QEngineOCLMultiPtr> toCopy) {
    std::map<QInterfacePtr, bitLenInt> ret;
    
    for (auto&& q : toCopy) {
        ret[q] = Cohere(q);
    }
    
    return ret;
}
    
void QEngineOCLMulti::Decohere(bitLenInt start, bitLenInt length, QEngineOCLMultiPtr dest) {
    CombineAllEngines();
    dest->CombineAllEngines();
    substateEngines[0]->Decohere(start, length, dest->substateEngines[0]);
    SetQubitCount(qubitCount - length);
    SeparateAllEngines();
    dest->SeparateAllEngines();
}

void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length) {
    CombineAllEngines();
    substateEngines[0]->Dispose(start, length);
    SetQubitCount(qubitCount - length);
    SeparateAllEngines();
}
    
#if 0
void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length) {
    
    if (subEngineCount == 1) {
        substateEngines[0]->Dispose(start, length);
        return;
    }
    
    int i;
    
    // Move the bits to be disposed to the end of the register, then reverse the "Cohere" operation.
    bitLenInt shiftForward = qubitCount - (start + length);
    if (shiftForward != 0) {
        ROL(shiftForward, 0, qubitCount);
    }
    
    // We can have a maximum of 2^N nodes for N qubits
    if (subEngineCount > (1 << (qubitCount - length))) {
        bitLenInt nSubEngineCount = 1 << (qubitCount - length);
        std::vector<QEngineOCLPtr> nSubstateEngines(nSubEngineCount);
        std::vector<std::future<void>> combineFutures(nSubEngineCount);
        bitLenInt maxLcv = subEngineCount / nSubEngineCount;
        bitLenInt nSubQubitCount = qubitCount / log2(nSubEngineCount);
        for (i = 0; i < nSubEngineCount; i++) {
            combineFutures[i] = std::async(std::launch::async, [this, i, maxLcv, nSubQubitCount, &nSubstateEngines]() {
                nSubstateEngines[i] = std::make_shared<QEngineOCL>(nSubQubitCount, 0, rand_generator, i, true);
                nSubstateEngines[i]->EnableNormalize(false);
                CommandQueuePtr queue = nSubstateEngines[i]->GetQueuePtr();
                for (int j = 0; j < maxLcv; j++) {
                    queue->enqueueCopyBuffer(
                        substateEngines[j + (i * maxLcv)]->GetStateBuffer(),
                        nSubstateEngines[i]->GetStateBuffer(),
                        0, j * (subBufferSize << 1), subBufferSize << 1);
                    queue->flush();
                    queue->finish();
                }
            });
        }
        for (i = 0; i < subEngineCount; i++) {
            combineFutures[i].get();
        }
        
        substateEngines = nSubstateEngines;
        subEngineCount = substateEngines.size();
        subQubitCount = qubitCount - log2(subEngineCount);
    }
    
    int destIndex, destEngine, sourceIndex, sourceEngine;
    
    cl::Context context = *(clObj->GetContextPtr());
    CommandQueuePtr queue;
    
    size_t divCount = 1 << length;
    size_t divSize = (sizeof(complex) * (1 << subQubitCount)) / divCount;
    
    std::vector<QEngineOCLPtr> tempEngines(subEngineCount);
    
    for (i = 0; i < subEngineCount; i++) {
        tempEngines[i] = std::make_shared<QEngineOCL>(substateEngines[i]);
    }
    
    for (i = 0; i < (subEngineCount * divCount); i++) {
        sourceEngine = (i / divCount);
        sourceIndex = i & (divCount - 1) ;
        destEngine = i & (subEngineCount - 1);
        destIndex = i / subEngineCount;
        
        queue = substateEngines[destEngine]->GetQueuePtr();
        
        queue->enqueueCopyBuffer(
            tempEngines[sourceEngine]->GetStateBuffer(),
            substateEngines[destEngine]->GetStateBuffer(),
            sourceIndex * divSize, destIndex * divSize, divSize);
        queue->flush();
        queue->finish();
    }
    
    std::vector<std::future<void>> futures(subEngineCount);
    for (i = 0; i < subEngineCount; i++) {
        futures[i] = std::async(std::launch::async, [this, i, length]() {
            substateEngines[i]->Dispose(subQubitCount - length, length);
        });
    }
    
    for (i = 0; i < subEngineCount; i++) {
        futures[i].get();
    }
    
    SetQubitCount(qubitCount - length);
    
    if (shiftForward != 0) {
        ROR(shiftForward, 0, qubitCount);
    }
}
#endif
    
void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
    DoublyControlledGate(false, control1, control2, target, (CCGFn)(&QEngineOCL::CCNOT), (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
}

void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
    DoublyControlledGate(true, control1, control2, target, (CCGFn)(&QEngineOCL::AntiCCNOT), (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target) {
    ControlledGate(true, control, target, (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::H(bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (GFn)(&QEngineOCL::H));
}
    
bool QEngineOCLMulti::M(bitLenInt qubit) {
    
    if (subEngineCount == 1) {
        return substateEngines[0]->M(qubit);
    }
    
    if (runningNorm != 1.0) {
        NormalizeState();
    }
    
    int i, j;
    
    real1 prob = Rand();
    real1 oneChance = Prob(qubit);
    
    bool result = ((prob < oneChance) && (oneChance > 0.0));
    real1 nrmlzr = 1.0;
    if (result) {
        nrmlzr = oneChance;
    } else {
        nrmlzr = 1.0 - oneChance;
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
        bitLenInt groupCount = 1<<(qubitCount - (qubit + 1));
        bitLenInt groupSize = 1 << ((qubit + 1) - subQubitCount);
        bitLenInt keepOffset, clearOffset;
        bitLenInt keepIndex, clearIndex;
        if (result) {
            keepOffset = 1;
            clearOffset = 0;
        }
        else {
            keepOffset = 0;
            clearOffset = 1;
        }
        for (i = 0; i < groupCount; i++) {
            for (j = 0; j < (groupSize / 2); j++) {
                clearIndex = j + (i * groupSize) + (clearOffset * groupSize / 2);
                keepIndex = j + (i * groupSize) + (keepOffset * groupSize / 2);
                
                cl::Buffer buffer = substateEngines[clearIndex]->GetStateBuffer();
                CommandQueuePtr queue = substateEngines[clearIndex]->GetQueuePtr();
                queue->enqueueFillBuffer(buffer, complex(0.0, 0.0), 0, subBufferSize << 1);
                queue->flush();
                
                substateEngines[keepIndex]->NormalizeState(nrmlzr);
            }
        }
    }
    
    return result;
}
    
void QEngineOCLMulti::X(bitLenInt qubitIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->X(qubitIndex);
    }, {qubitIndex});
    //SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::Y(bitLenInt qubitIndex) {
    SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Y));
}
    
void QEngineOCLMulti::Z(bitLenInt qubitIndex) {
    SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Z));
}
    
void QEngineOCLMulti::CY(bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CY), (GFn)(&QEngineOCL::Y));
}
    
void QEngineOCLMulti::CZ(bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CZ), (GFn)(&QEngineOCL::Z));
}
    
void QEngineOCLMulti::RT(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RT), radians);
}
void QEngineOCLMulti::RX(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::CRX(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRX), (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::RY(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::CRY(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRY), (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::RZ(real1 radians, bitLenInt qubitIndex) {
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRZ), (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRT(real1 radians, bitLenInt control, bitLenInt target) {
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRT), (RGFn)(&QEngineOCL::RT), radians);
}
    
void QEngineOCLMulti::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INC(toAdd, start, length);
    }, {static_cast<bitLenInt>(start + length - 1)});
}
void QEngineOCLMulti::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCC(toAdd, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
void QEngineOCLMulti::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCS(toAdd, start, length, overflowIndex);
    }, {static_cast<bitLenInt>(start + length - 1), overflowIndex});
}
void QEngineOCLMulti::INCSC(
                       bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), overflowIndex, carryIndex});
}
void QEngineOCLMulti::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCSC(toAdd, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
void QEngineOCLMulti::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCBCD(toAdd, start, length);
    }, {static_cast<bitLenInt>(start + length - 1)});
}
void QEngineOCLMulti::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
void QEngineOCLMulti::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DEC(toSub, start, length);
    }, {static_cast<bitLenInt>(start + length - 1)});
}
void QEngineOCLMulti::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECC(toSub, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
void QEngineOCLMulti::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECS(toSub, start, length, overflowIndex);
    }, {static_cast<bitLenInt>(start + length - 1), overflowIndex});
}
void QEngineOCLMulti::DECSC(
                       bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), overflowIndex, carryIndex});
}
void QEngineOCLMulti::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECSC(toSub, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
void QEngineOCLMulti::DECBCD(bitCapInt toSub, bitLenInt start, bitLenInt length) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECBCD(toSub, start, length);
    }, {static_cast<bitLenInt>(start + length - 1)});
}
void QEngineOCLMulti::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->DECBCDC(toSub, start, length, carryIndex);
    }, {static_cast<bitLenInt>(start + length - 1), carryIndex});
}
    
void QEngineOCLMulti::ZeroPhaseFlip(bitLenInt start, bitLenInt length) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->ZeroPhaseFlip(start, length);
    }, {static_cast<bitLenInt>(start + length - 1)});
}
void QEngineOCLMulti::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }, {static_cast<bitLenInt>(start + length - 1), flagIndex});
}
void QEngineOCLMulti::PhaseFlip() {
    for (bitLenInt i = 0; i < subEngineCount; i++) {
        substateEngines[i]->PhaseFlip();
    }
}
    
void QEngineOCLMulti::X(bitLenInt start, bitLenInt length) {
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->X(start, len);
    }, [this](bitLenInt bit) {
        X(bit);
    }, start, length);
}

void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target, bitLenInt length) {
    ControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->CNOT(control, target, len);
    }, [this](bitLenInt control, bitLenInt target) {
        QEngineOCLMulti::CNOT(control, target);
    }, control, target, length);
}

void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length) {
    ControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->AntiCNOT(control, target, len);
    }, [this](bitLenInt control, bitLenInt target) {
        QEngineOCLMulti::AntiCNOT(control, target);
    }, control, target, length);
}
    
void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length) {
    DoublyControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->CCNOT(control1, control2, target, len);
    }, [this](bitLenInt control1, bitLenInt control2, bitLenInt target) {
        QEngineOCLMulti::CCNOT(control1, control2, target);
    }, control1, control2, target, length);
}
    
void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length) {
    DoublyControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->AntiCCNOT(control1, control2, target, len);
    }, [this](bitLenInt control1, bitLenInt control2, bitLenInt target) {
        QEngineOCLMulti::AntiCCNOT(control1, control2, target);
    }, control1, control2, target, length);
}
    
#if 0
void QEngineOCLMulti::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length) {
    DoublyControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->AND(inputBit1, inputBit2, outputBit, len);
    }, [this](bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
        QInterface::AND(inputBit1, inputBit2, outputBit);
    }, inputBit1, inputBit2, outputBit, length);
}
    
void QEngineOCLMulti::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length) {
    DoublyControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->OR(inputBit1, inputBit2, outputBit, len);
    }, [this](bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
        QInterface::OR(inputBit1, inputBit2, outputBit);
    }, inputBit1, inputBit2, outputBit, length);
}
    
void QEngineOCLMulti::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length) {
    DoublyControlledRegOp([&](QEngineOCLPtr engine, bitLenInt len) {
        engine->XOR(inputBit1, inputBit2, outputBit, len);
    }, [this](bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
        QInterface::XOR(inputBit1, inputBit2, outputBit);
    }, inputBit1, inputBit2, outputBit, length);
}
#endif
    
bitCapInt QEngineOCLMulti::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, unsigned char* values) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values);
    }, {static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1)});
    
    return 0;
}
    
bitCapInt QEngineOCLMulti::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }, {static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1), carryIndex});
    
    return 0;
}
bitCapInt QEngineOCLMulti::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
                                 bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) {
    CombineAndOp([&](QEngineOCLPtr engine) {
        engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }, {static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1), carryIndex});
    
    return 0;
}
    
void QEngineOCLMulti::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) {
    
    if (subEngineCount == 1) {
        substateEngines[0]->Swap(qubitIndex1, qubitIndex2);
        return;
    }
    
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
        SetQubitCount(qubitCount);
    } else {
        // "Swap" is tricky, if we're distributed across nodes.
        // However, we get it virtually for free in a QUnit, so this is a low-priority case.
        // Assuming our CNOT works, so does this:
        CNOT(qubitIndex1, qubitIndex2);
        CNOT(qubitIndex2, qubitIndex1);
        CNOT(qubitIndex1, qubitIndex2);
    }
}
void QEngineOCLMulti::CopyState(QEngineOCLMultiPtr orig) {
    CombineAllEngines();
    orig->CombineAllEngines();
    substateEngines[0]->CopyState(orig->substateEngines[0]);
    SeparateAllEngines();
    orig->SeparateAllEngines();
}
real1 QEngineOCLMulti::Prob(bitLenInt qubitIndex) {
    
    if (subEngineCount == 1) {
        return substateEngines[0]->Prob(qubitIndex);
    }

    real1 oneChance = 0.0;
    int i, j, k;

    if (qubitIndex < subQubitCount) {
        std::vector<std::future<real1>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
        }
        for (i = 0; i < subEngineCount; i++) {
            oneChance += futures[i].get();
        }
    } else {
        std::vector<std::future<real1>> futures(subEngineCount / 2);
        bitLenInt groupCount = 1<<(qubitCount - (qubitIndex + 1));
        bitLenInt groupSize = 1 << ((qubitIndex + 1) - subQubitCount);
        k = 0;
        for (i = 0; i < groupCount; i++) {
            for (j = 0; j < (groupSize / 2); j++) {
                QEngineOCLPtr engine = substateEngines[j + (i * groupSize) + (groupSize / 2)];
                futures[k] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->GetNorm(); });
                k++;
            }
        }
        
        for (i = 0; i < k; i++) {
            oneChance += futures[i].get();
        }
    }
    
    return oneChance;
}
real1 QEngineOCLMulti::ProbAll(bitCapInt fullRegister) {
    bitLenInt subIndex = fullRegister / subMaxQPower;
    fullRegister -= subIndex * subMaxQPower;
    //if (isnan(substateEngines[subIndex]->ProbAll(fullRegister))) {
    //    std::cout<<"isNaN: subIndex="<<(int)subIndex<<" fullRegister="<<(int)fullRegister<<std::endl;
    //}
    return substateEngines[subIndex]->ProbAll(fullRegister);
}
    
template<typename CF, typename F, typename ... Args> void QEngineOCLMulti::ControlledBody(bool anti, bitLenInt controlDepth, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args ... gfnArgs) {
    int i, j, k;
    
    std::vector<std::future<void>> futures(subEngineCount / 2);
    cl::Context context = *(clObj->GetContextPtr());
    
    bitLenInt order, groups, diffOrder, pairOffset, groupOffset;
    bool useTopDevice;
    
    if (targetBit < controlBit) {
        int pairPower = subEngineCount - (1 << (qubitCount - (targetBit + 1)));
        
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
                futures[k + (j * groupOffset) + (i * groupOffset * pairOffset)] = std::async(std::launch::async, [this, anti, useTopDevice, context, pairOffset, index, fn, sqc, gfnArgs ...]() {
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
                        if (anti) {
                            engine = substateEngines[index];
                        }
                        else {
                            engine = substateEngines[index + pairOffset];
                        }
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
    
// For scalable cluster distribution, these methods should ultimately be entirely removed:
void QEngineOCLMulti::CombineAllEngines() {
    
    if (subEngineCount == 1) {
        return;
    }
    
    QEngineOCLPtr nEngine = std::make_shared<QEngineOCL>(qubitCount, 0, rand_generator, 0);
    
    CommandQueuePtr queue;
    size_t sbSize = sizeof(complex) * maxQPower / subEngineCount;
    
    for (bitLenInt i = 0; i < subEngineCount; i++) {
        queue = substateEngines[i]->GetQueuePtr();
        queue->enqueueCopyBuffer(
            substateEngines[i]->GetStateBuffer(),
            nEngine->GetStateBuffer(),
            0, i * sbSize, sbSize);
        queue->flush();
        queue->finish();
    }
    
    substateEngines.resize(1);
    substateEngines[0] = nEngine;
    SetQubitCount(qubitCount);
}
    
void QEngineOCLMulti::SeparateAllEngines() {
    bitLenInt engineCount = 1 << maxDeviceOrder;
    
    if (maxDeviceOrder >= qubitCount) {
        engineCount = 1 << (qubitCount - 1);
    }
    
    if (engineCount == 1) {
        return;
    }
    
    bitLenInt i;
    
    std::vector<QEngineOCLPtr> nSubEngines(engineCount);
    
    CommandQueuePtr queue;
    size_t sbSize = sizeof(complex) * (1 << qubitCount) / engineCount;
    
    for (i = 0; i < engineCount; i++) {
        nSubEngines[i] = std::make_shared<QEngineOCL>(qubitCount - log2(engineCount), 0, rand_generator, i, true);
        nSubEngines[i]->EnableNormalize(false);
        queue = nSubEngines[i]->GetQueuePtr();
        queue->enqueueCopyBuffer(
            substateEngines[0]->GetStateBuffer(),
            nSubEngines[i]->GetStateBuffer(),
            i * sbSize, 0, sbSize);
        queue->flush();
        queue->finish();
    }
    
    substateEngines = nSubEngines;
    SetQubitCount(qubitCount);
}
    
template <typename F> void QEngineOCLMulti::CombineAndOp(F fn, std::vector<bitLenInt> bits) {
    bitLenInt i;
    bitLenInt highestBit = 0;
    for (i = 0; i < bits.size(); i++) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
    }
        
    if (highestBit < subQubitCount) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            futures[i] = std::async(std::launch::async, [this, fn, i]() { fn(substateEngines[i]); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    }
    else {
        CombineAllEngines();
        fn(substateEngines[0]);
        SeparateAllEngines();
    }
}
    
template <typename F, typename OF> void QEngineOCLMulti::RegOp(F fn, OF ofn, bitLenInt start, bitLenInt length) {
    
    if (subEngineCount == 1) {
        fn(substateEngines[0], length);
    }
    
    bitLenInt i;
    bitLenInt highestBit = start + length - 1;
    
    std::vector<std::future<void>> futures(subEngineCount);
    if (highestBit < subQubitCount) {
        for (i = 0; i < subEngineCount; i++) {
            futures[i] = std::async(std::launch::async, [this, fn, i, length]() { fn(substateEngines[i], length); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        bitLenInt bitDiff = (highestBit - subQubitCount) + 1;
        int subLength = length - bitDiff;
        if (subLength > 0) {
            for (i = 0; i < subEngineCount; i++) {
                futures[i] = std::async(std::launch::async, [this, fn, i, subLength]() { fn(substateEngines[i], subLength); });
            }
            for (i = 0; i < subEngineCount; i++) {
                futures[i].get();
            }
        }
        else {
            subLength = 0;
        }
        for (i = subLength; i < length; i++) {
            ofn(start + i);
        }
    }
}
            
template <typename F, typename OF> void QEngineOCLMulti::ControlledRegOp(F fn, OF ofn, bitLenInt control, bitLenInt target, bitLenInt length) {
    
    if (subEngineCount == 1) {
        fn(substateEngines[0], length);
    }

    bitLenInt i;
    bitLenInt highestBit;
    if (target > control) {
        highestBit = target + length - 1;
    } else {
        highestBit = control + length - 1;
    }
    
    std::vector<std::future<void>> futures(subEngineCount);
    if (highestBit < subQubitCount) {
        for (i = 0; i < subEngineCount; i++) {
            futures[i] = std::async(std::launch::async, [this, fn, i, length]() { fn(substateEngines[i], length); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        bitLenInt bitDiff = (highestBit - subQubitCount) + 1;
        int subLength = length - bitDiff;
        if (subLength > 0) {
            for (i = 0; i < subEngineCount; i++) {
                futures[i] = std::async(std::launch::async, [this, fn, i, subLength]() { fn(substateEngines[i], subLength); });
            }
            for (i = 0; i < subEngineCount; i++) {
                futures[i].get();
            }
        }
        else {
            subLength = 0;
        }
        for (i = subLength; i < length; i++) {
            ofn(control + i, target + i);
        }
    }
}
    
template <typename F, typename OF> void QEngineOCLMulti::DoublyControlledRegOp(F fn, OF ofn, bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length) {
    
    if (subEngineCount == 1) {
        fn(substateEngines[0], length);
    }
    
    bitLenInt i;
    bitLenInt highestBit;
    if ((target >= control1) && (target >= control2)) {
        highestBit = target + length - 1;
    } else if ((control1 >= target) && (control1 >= control2)) {
        highestBit = control1 + length - 1;
    }
    else {
        highestBit = control2 + length - 1;
    }
    
    std::vector<std::future<void>> futures(subEngineCount);
    if (highestBit < subQubitCount) {
        for (i = 0; i < subEngineCount; i++) {
            futures[i] = std::async(std::launch::async, [this, fn, i, length]() { fn(substateEngines[i], length); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        bitLenInt bitDiff = (highestBit - subQubitCount) + 1;
        int subLength = length - bitDiff;
        if (subLength > 0) {
            for (i = 0; i < subEngineCount; i++) {
                futures[i] = std::async(std::launch::async, [this, fn, i, subLength]() { fn(substateEngines[i], subLength); });
            }
            for (i = 0; i < subEngineCount; i++) {
                futures[i].get();
            }
        }
        else {
            subLength = 0;
        }
        for (i = subLength; i < length; i++) {
            ofn(control1 + i, control2 + i, target + i);
        }
    }
}

void QEngineOCLMulti::NormalizeState() {
    bitLenInt i;
    if (runningNorm <= 0.0) {
        runningNorm = 0.0;
        for (i = 0; i < subEngineCount; i++) {
            runningNorm += substateEngines[i]->GetNorm();
        }
    }
    
    if ((runningNorm > 0.0) && (runningNorm != 1.0)) {
        for (i = 0; i < subEngineCount; i++) {
            substateEngines[i]->NormalizeState(runningNorm);
        }
    }
    
    runningNorm = 1.0;
}
    
} // namespace Qrack
