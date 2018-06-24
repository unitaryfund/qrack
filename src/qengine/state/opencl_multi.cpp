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

QEngineOCLMulti::QEngineOCLMulti(
    bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, int deviceCount)
    : QInterface(qBitCount)
{
    
    rand_generator = rgp;

    runningNorm = 1.0;

    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetDeviceCount();
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

    if (deviceCount == 1) {
        substateEngines.push_back(std::make_shared<QEngineOCL>(qubitCount, initState, rgp));
        substateEngines[0]->EnableNormalize(true);
        return;
    }

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

void QEngineOCLMulti::ShuffleBuffers(CommandQueuePtr queue, BufferPtr buff1, BufferPtr buff2, BufferPtr tempBuffer)
{
    queue->enqueueCopyBuffer(*buff1, *tempBuffer, subBufferSize, 0, subBufferSize);
    queue->flush();
    queue->finish();

    queue->enqueueCopyBuffer(*buff2, *buff1, 0, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();

    queue->enqueueCopyBuffer(*tempBuffer, *buff2, 0, 0, subBufferSize);
    queue->flush();
    queue->finish();
}

void QEngineOCLMulti::SwapBuffersLow(CommandQueuePtr queue, BufferPtr buff1, BufferPtr buff2, BufferPtr tempBuffer)
{
    queue->enqueueCopyBuffer(*buff1, *tempBuffer, subBufferSize, 0, subBufferSize);
    queue->flush();
    queue->finish();

    queue->enqueueCopyBuffer(*buff2, *buff1, subBufferSize, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();

    queue->enqueueCopyBuffer(*tempBuffer, *buff2, 0, subBufferSize, subBufferSize);
    queue->flush();
    queue->finish();
}

template <typename F, typename... Args>
void QEngineOCLMulti::SingleBitGate(bool controlled, bool anti, bool doNormalize, bitLenInt bit, F fn, Args... gfnArgs)
{

    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*fn)(gfnArgs..., bit);
        return;
    }
    
    bitLenInt i, j;
    
    if (runningNorm != 1.0) {
        for (i = 0; i < subEngineCount; i++) {
            substateEngines[i]->SetNorm(runningNorm);
            substateEngines[i]->EnableNormalize(true);
        }
        runningNorm = 1.0;
    }
    else if (doNormalize) {
        for (i = 0; i < subEngineCount; i++) {
            substateEngines[i]->SetNorm(1.0);
            substateEngines[i]->EnableNormalize(true);
        }
    }
    
    if (bit < subQubitCount) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(
                std::launch::async, [engine, fn, bit, gfnArgs...]() { ((engine.get())->*fn)(gfnArgs..., bit); });
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

        bitLenInt sqi = subQubitCount - 1;

        for (i = 0; i < groups; i++) {
            for (j = 0; j < offset; j++) {
                futures[j + (i * offset)] = std::async(
                    std::launch::async, [this, context, offset, i, j, fn, sqi, controlled, anti, gfnArgs...]() {
                        QEngineOCLPtr engine1 = substateEngines[j + (i * offset)];
                        QEngineOCLPtr engine2 = substateEngines[j + ((i + 1) * offset)];
                        CommandQueuePtr queue = engine1->GetQueuePtr();

                        BufferPtr tempBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, subBufferSize);
                        BufferPtr buff1 = engine1->GetStateBufferPtr();
                        BufferPtr buff2 = engine2->GetStateBufferPtr();

                        ShuffleBuffers(queue, buff1, buff2, tempBuffer);

                        if (controlled) {
                            if (anti) {
                                ((engine1.get())->*fn)(gfnArgs..., sqi);
                            } else {
                                ((engine2.get())->*fn)(gfnArgs..., sqi);
                            }
                        } else {
                            std::future<void> future1 = std::async(std::launch::async,
                                [engine1, fn, sqi, gfnArgs...]() { ((engine1.get())->*fn)(gfnArgs..., sqi); });
                            std::future<void> future2 = std::async(std::launch::async,
                                [engine2, fn, sqi, gfnArgs...]() { ((engine2.get())->*fn)(gfnArgs..., sqi); });
                            future1.get();
                            future2.get();
                        }

                        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
                    });
            }
        }
        for (i = 0; i < subEngineCount / 2; i++) {
            futures[i].get();
        }
    }

    if (doNormalize) {
        std::vector<std::future<real1>> nf(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            nf[i] = std::async(std::launch::async, [this, i]() {
                return substateEngines[i]->GetNorm();
            });
        }
        runningNorm = 0.0;
        for (i = 0; i <subEngineCount; i++) {
            runningNorm += nf[i].get();
        }
    }
    
    for (i = 0; i < subEngineCount; i++) {
        substateEngines[i]->EnableNormalize(false);
    }
}

template <typename CF, typename F, typename... Args>
void QEngineOCLMulti::ControlledGate(
    bool anti, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args... gfnArgs)
{

    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*cfn)(gfnArgs..., controlBit, targetBit);
        return;
    }

    bitLenInt i;

    if ((controlBit < subQubitCount) && (targetBit < subQubitCount)) {
        std::vector<std::future<void>> futures(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(std::launch::async, [engine, cfn, controlBit, targetBit, gfnArgs...]() {
                ((engine.get())->*cfn)(gfnArgs..., controlBit, targetBit);
            });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else if (targetBit < controlBit) {
        std::vector<std::future<void>> futures(subEngineCount / 2);
        bitLenInt offset = 1 << (qubitCount - (controlBit + 1));
        bitLenInt antiFactor = anti ? 0 : 1;
        for (i = 0; i < (subEngineCount / 2); i++) {
            QEngineOCLPtr engine = substateEngines[(i * offset) + (subEngineCount / (2 * offset)) * antiFactor];
            futures[i] = std::async(std::launch::async,
                [engine, fn, targetBit, gfnArgs...]() { ((engine.get())->*fn)(gfnArgs..., targetBit); });
        }
        for (i = 0; i < subEngineCount / 2; i++) {
            futures[i].get();
        }
    } else {
        CombineAndOp([&](QEngineOCLPtr engine) { (engine.get()->*cfn)(gfnArgs..., controlBit, targetBit); },
            { controlBit, targetBit });
    }
}

template <typename CCF, typename CF, typename F, typename... Args>
void QEngineOCLMulti::DoublyControlledGate(bool anti, bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt targetBit,
    CCF ccfn, CF cfn, F fn, Args... gfnArgs)
{
    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*ccfn)(gfnArgs..., controlBit1, controlBit2, targetBit);
        return;
    }

    bitLenInt lowControl, highControl;
    if (controlBit1 < controlBit2) {
        lowControl = controlBit1;
        highControl = controlBit2;
    } else {
        lowControl = controlBit2;
        highControl = controlBit1;
    }

    if (lowControl < subQubitCount) {
        // CNOT logic
        ControlledGate(anti, highControl, targetBit, ccfn, cfn, gfnArgs..., lowControl);
    } else {
        // Skip first group, if more than one group.
        CombineAndOp(
            [&](QEngineOCLPtr engine) { (engine.get()->*ccfn)(gfnArgs..., controlBit1, controlBit2, targetBit); },
            { controlBit1, controlBit2, targetBit });
    }
}

void QEngineOCLMulti::SetQuantumState(complex* inputState)
{
    CombineAllEngines();
    substateEngines[0]->SetQuantumState(inputState);
    SeparateAllEngines();
}

void QEngineOCLMulti::SetPermutation(bitCapInt perm)
{
    if (subEngineCount == 1) {
        substateEngines[0]->SetPermutation(perm);
        return;
    }

    std::future<void> ftr;
    bitCapInt i;
    bitCapInt j = 0;
    for (i = 0; i < maxQPower; i += subMaxQPower) {
        if ((perm >= i) && (perm < (i + subMaxQPower))) {
            QEngineOCLPtr engine = substateEngines[j];
            bitCapInt p = perm - i;
            ftr = std::async(std::launch::async, [engine, p]() { engine->SetPermutation(p); });
        } else {
            BufferPtr buffer = substateEngines[j]->GetStateBufferPtr();
            CommandQueuePtr queue = substateEngines[j]->GetQueuePtr();
            queue->enqueueFillBuffer(*buffer, complex(0.0, 0.0), 0, subBufferSize << 1);
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

bitLenInt QEngineOCLMulti::Cohere(QEngineOCLMultiPtr toCopy)
{
    bitLenInt result;
    CombineAllEngines();
    toCopy->CombineAllEngines();
    result = substateEngines[0]->Cohere(toCopy->substateEngines[0]);
    SetQubitCount(qubitCount + toCopy->qubitCount);
    SeparateAllEngines();
    toCopy->SeparateAllEngines();
    return result;
}

std::map<QInterfacePtr, bitLenInt> QEngineOCLMulti::Cohere(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto&& q : toCopy) {
        ret[q] = Cohere(q);
    }

    return ret;
}

void QEngineOCLMulti::Decohere(bitLenInt start, bitLenInt length, QEngineOCLMultiPtr dest)
{
    CombineAllEngines();
    dest->CombineAllEngines();
    substateEngines[0]->Decohere(start, length, dest->substateEngines[0]);
    if (qubitCount <= length) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }
    SeparateAllEngines();
    dest->SeparateAllEngines();
}

void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length)
{
    CombineAllEngines();
    substateEngines[0]->Dispose(start, length);
    if (qubitCount <= length) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }
    SeparateAllEngines();
}

void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    DoublyControlledGate(false, control1, control2, target, (CCGFn)(&QEngineOCL::CCNOT), (CGFn)(&QEngineOCL::CNOT),
        (GFn)(&QEngineOCL::X));
}

void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
}

void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    DoublyControlledGate(true, control1, control2, target, (CCGFn)(&QEngineOCL::AntiCCNOT),
        (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
}

void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target)
{
    ControlledGate(true, control, target, (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
}
    
void QEngineOCLMulti::H(bitLenInt qubitIndex) { SingleBitGate(false, false, true, qubitIndex, (GFn)(&QEngineOCL::H)); }

bool QEngineOCLMulti::M(bitLenInt qubit)
{

    if (subEngineCount == 1) {
        return substateEngines[0]->M(qubit);
    }

    NormalizeState();

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
            futures[i] = std::async(
                std::launch::async, [engine, qubit, result, nrmlzr]() { engine->ForceM(qubit, result, true, nrmlzr); });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        bitLenInt groupCount = 1 << (qubitCount - (qubit + 1));
        bitLenInt groupSize = 1 << ((qubit + 1) - subQubitCount);
        bitLenInt keepOffset, clearOffset;
        bitLenInt keepIndex, clearIndex;
        if (result) {
            keepOffset = 1;
            clearOffset = 0;
        } else {
            keepOffset = 0;
            clearOffset = 1;
        }
        for (i = 0; i < groupCount; i++) {
            for (j = 0; j < (groupSize / 2); j++) {
                clearIndex = j + (i * groupSize) + (clearOffset * groupSize / 2);
                keepIndex = j + (i * groupSize) + (keepOffset * groupSize / 2);

                BufferPtr buffer = substateEngines[clearIndex]->GetStateBufferPtr();
                CommandQueuePtr queue = substateEngines[clearIndex]->GetQueuePtr();
                queue->enqueueFillBuffer(*buffer, complex(0.0, 0.0), 0, subBufferSize << 1);
                queue->flush();

                if (nrmlzr > 0.0) {
                    substateEngines[keepIndex]->NormalizeState(nrmlzr);
                }
            }
        }
    }

    return result;
}

void QEngineOCLMulti::X(bitLenInt qubitIndex) { SingleBitGate(false, false, false, qubitIndex, (GFn)(&QEngineOCL::X)); }

void QEngineOCLMulti::Y(bitLenInt qubitIndex) { SingleBitGate(false, false, false, qubitIndex, (GFn)(&QEngineOCL::Y)); }

void QEngineOCLMulti::Z(bitLenInt qubitIndex) { SingleBitGate(false, false, false, qubitIndex, (GFn)(&QEngineOCL::Z)); }

void QEngineOCLMulti::CY(bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CY), (GFn)(&QEngineOCL::Y));
}

void QEngineOCLMulti::CZ(bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CZ), (GFn)(&QEngineOCL::Z));
}

void QEngineOCLMulti::RT(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(false, false, true, qubitIndex, (RGFn)(&QEngineOCL::RT), radians);
}
void QEngineOCLMulti::RX(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(false, false, true, qubitIndex, (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::CRX(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRX), (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::RY(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(false, false, true, qubitIndex, (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::CRY(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRY), (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::RZ(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(false, false, true, qubitIndex, (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRZ), (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRT(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRT), (RGFn)(&QEngineOCL::RT), radians);
}

void QEngineOCLMulti::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INC(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1) });
}
void QEngineOCLMulti::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}
void QEngineOCLMulti::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCS(toAdd, start, length, overflowIndex); },
        { static_cast<bitLenInt>(start + length - 1), overflowIndex });
}
void QEngineOCLMulti::INCSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCSC(toAdd, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), overflowIndex, carryIndex });
}
void QEngineOCLMulti::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCSC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}
void QEngineOCLMulti::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCBCD(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1) });
}
void QEngineOCLMulti::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->INCBCDC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}
void QEngineOCLMulti::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DEC(toSub, start, length); },
        { static_cast<bitLenInt>(start + length - 1) });
}
void QEngineOCLMulti::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}
void QEngineOCLMulti::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECS(toSub, start, length, overflowIndex); },
        { static_cast<bitLenInt>(start + length - 1), overflowIndex });
}
void QEngineOCLMulti::DECSC(
    bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECSC(toSub, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), overflowIndex, carryIndex });
}
void QEngineOCLMulti::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECSC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}
void QEngineOCLMulti::DECBCD(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECBCD(toSub, start, length); },
        { static_cast<bitLenInt>(start + length - 1) });
}
void QEngineOCLMulti::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DECBCDC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1), carryIndex });
}

void QEngineOCLMulti::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->ZeroPhaseFlip(start, length); },
        { static_cast<bitLenInt>(start + length - 1) });
}
void QEngineOCLMulti::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); },
        { static_cast<bitLenInt>(start + length - 1), flagIndex });
}
void QEngineOCLMulti::PhaseFlip()
{
    for (bitLenInt i = 0; i < subEngineCount; i++) {
        substateEngines[i]->PhaseFlip();
    }
}
    
void QEngineOCLMulti::X(bitLenInt start, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->X(start, len); }, [&](bitLenInt offset) { X(start + offset); }, length, { static_cast<bitLenInt>(start + length - 1) });
}

void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->CNOT(control, target, len); }, [&](bitLenInt offset) { CNOT(control + offset, target + offset); }, length, { static_cast<bitLenInt>(control + length - 1), static_cast<bitLenInt>(target + length - 1) });
}

void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->AntiCNOT(control, target, len); }, [&](bitLenInt offset) { AntiCNOT(control + offset, target + offset); }, length, { static_cast<bitLenInt>(control + length - 1), static_cast<bitLenInt>(target + length - 1) });
}

void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->CCNOT(control1, control2, target, len); }, [&](bitLenInt offset) { CCNOT(control1 + offset, control2 + offset, target + offset); }, length, { static_cast<bitLenInt>(control1 + length - 1), static_cast<bitLenInt>(control2 + length - 1), static_cast<bitLenInt>(target + length - 1) });
}

void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->AntiCCNOT(control1, control2, target, len); }, [&](bitLenInt offset) { AntiCCNOT(control1 + offset, control2 + offset, target + offset); }, length, { static_cast<bitLenInt>(control1 + length - 1), static_cast<bitLenInt>(control2 + length - 1), static_cast<bitLenInt>(target + length - 1) });
}

void QEngineOCLMulti::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, 0);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit, length);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

void QEngineOCLMulti::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, (1 << length) - 1);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit, length);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

void QEngineOCLMulti::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetReg(outputBit, length, 0);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit, length);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit, length);
    } else {
        SetReg(outputBit, length, 0);
        CNOT(inputBit1, outputBit, length);
        CNOT(inputBit2, outputBit, length);
    }
}

bitCapInt QEngineOCLMulti::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    CombineAndOpSafe(
        [&](QEngineOCLPtr engine) { engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values); },
        { static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1) });

    return 0;
}

bitCapInt QEngineOCLMulti::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOpSafe(
        [&](QEngineOCLPtr engine) {
            engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1),
            carryIndex });

    return 0;
}
bitCapInt QEngineOCLMulti::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOpSafe(
        [&](QEngineOCLPtr engine) {
            engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1),
            carryIndex });

    return 0;
}

void QEngineOCLMulti::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{

    if (qubitIndex1 == qubitIndex2) {
        return;
    }

    if (subEngineCount == 1) {
        substateEngines[0]->Swap(qubitIndex1, qubitIndex2);
        return;
    }

    if ((qubitIndex1 < subQubitCount) && (qubitIndex2 < subQubitCount)) {
        // Here, it's entirely contained within single nodes:
        std::vector<std::future<void>> futures(subEngineCount);
        bitLenInt i;
        for (i = 0; i < subEngineCount; i++) {
            QEngineOCLPtr engine = substateEngines[i];
            futures[i] = std::async(
                std::launch::async, [engine, qubitIndex1, qubitIndex2]() { engine->Swap(qubitIndex1, qubitIndex2); });
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

        for (bitLenInt i = 0; i < subEngineCount; i++) {
            substateEngines[i] = nSubstateEngines[i];
        }
        SetQubitCount(qubitCount);
    } else {
        // "Swap" is tricky, if we're distributed across nodes.
        // However, we get it virtually for free in a QUnit, so this is a low-priority case.
        // Assuming our CNOT works, so does this:
        // CNOT(qubitIndex1, qubitIndex2);
        // CNOT(qubitIndex2, qubitIndex1);
        // CNOT(qubitIndex1, qubitIndex2);

        // The above commented-out logic will work in general, but doesn't receive benefit from multiple device
        // parallelism with the current CNOT implementation. We might as well use this.
        CombineAndOp(
            [&](QEngineOCLPtr engine) { engine.get()->Swap(qubitIndex1, qubitIndex2); }, { qubitIndex1, qubitIndex2 });
    }
}
void QEngineOCLMulti::CopyState(QEngineOCLMultiPtr orig)
{
    CombineAllEngines();
    orig->CombineAllEngines();
    substateEngines[0]->CopyState(orig->substateEngines[0]);
    SeparateAllEngines();
    orig->SeparateAllEngines();
}
real1 QEngineOCLMulti::Prob(bitLenInt qubitIndex)
{
    if (subEngineCount == 1) {
        return substateEngines[0]->Prob(qubitIndex);
    }
    
    NormalizeState();

    real1 oneChance = 0.0;
    bitLenInt i, j, k;

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
        bitLenInt groupCount = 1 << (qubitCount - (qubitIndex + 1));
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
real1 QEngineOCLMulti::ProbAll(bitCapInt fullRegister)
{
    NormalizeState();
    
    bitLenInt subIndex = fullRegister / subMaxQPower;
    fullRegister -= subIndex * subMaxQPower;
    return substateEngines[subIndex]->ProbAll(fullRegister);
}

// For scalable cluster distribution, these methods should ultimately be entirely removed:
void QEngineOCLMulti::CombineAllEngines()
{

    if (subEngineCount == 1) {
        return;
    }

    QEngineOCLPtr nEngine = std::make_shared<QEngineOCL>(qubitCount, 0, rand_generator, 0);
    nEngine->EnableNormalize(true);

    CommandQueuePtr queue;
    size_t sbSize = sizeof(complex) * maxQPower / subEngineCount;

    for (bitLenInt i = 0; i < subEngineCount; i++) {
        queue = substateEngines[i]->GetQueuePtr();
        queue->enqueueCopyBuffer(
            *(substateEngines[i]->GetStateBufferPtr()), *(nEngine->GetStateBufferPtr()), 0, i * sbSize, sbSize);
        queue->flush();
        queue->finish();
    }

    substateEngines.resize(1);
    substateEngines[0] = nEngine;
    SetQubitCount(qubitCount);
}

void QEngineOCLMulti::SeparateAllEngines()
{

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
            *(substateEngines[0]->GetStateBufferPtr()), *(nSubEngines[i]->GetStateBufferPtr()), i * sbSize, 0, sbSize);
        queue->flush();
        queue->finish();
    }

    substateEngines.resize(engineCount);
    for (i = 0; i < engineCount; i++) {
        substateEngines[i] = nSubEngines[i];
    }
    SetQubitCount(qubitCount);
}

template <typename F> void QEngineOCLMulti::CombineAndOp(F fn, std::vector<bitLenInt> bits)
{
    if (subEngineCount == 1) {
        fn(substateEngines[0]);
        return;
    }

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
    } else {
        CombineAllEngines();
        fn(substateEngines[0]);
        SeparateAllEngines();
    }
}

template <typename F> void QEngineOCLMulti::CombineAndOpSafe(F fn, std::vector<bitLenInt> bits)
{
    if (subEngineCount == 1) {
        fn(substateEngines[0]);
        return;
    }

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
            futures[i] = std::async(std::launch::async, [this, fn, i]() {
                if (substateEngines[i]->GetNorm() > 0.0) {
                    fn(substateEngines[i]);
                }
            });
        }
        for (i = 0; i < subEngineCount; i++) {
            futures[i].get();
        }
    } else {
        CombineAllEngines();
        fn(substateEngines[0]);
        SeparateAllEngines();
    }
}

template <typename F, typename OF> void QEngineOCLMulti::RegOp(F fn, OF ofn, bitLenInt length, std::vector<bitLenInt> bits)
{
    if (subEngineCount == 1) {
        fn(substateEngines[0], length);
        return;
    }

    bitLenInt i;
    bitLenInt highestBit = 0;
    for (i = 0; i < bits.size(); i++) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
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
                futures[i] =
                std::async(std::launch::async, [this, fn, i, subLength]() { fn(substateEngines[i], subLength); });
            }
            for (i = 0; i < subEngineCount; i++) {
                futures[i].get();
            }
        } else {
            subLength = 0;
        }
        for (i = subLength; i < length; i++) {
            ofn(i);
        }
    }
}
    
void QEngineOCLMulti::NormalizeState()
{
    bitLenInt i;
    if (runningNorm != 1.0) {
        std::vector<std::future<void>> nf(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            nf[i] = std::async(std::launch::async, [this, i]() {
                substateEngines[i]->NormalizeState(runningNorm);
                substateEngines[i]->SetNorm(1.0);
            });
        }
        for (i = 0; i < subEngineCount; i++) {
            nf[i].get();
        }
    }
    runningNorm = 1.0;
}

} // namespace Qrack
