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
//
// QEngineOCLMulti allows distribution of a single QEngine on multiple OpenCL devices. Its algorithms are similar to
// those of the qHiPSTER project. (https://arxiv.org/abs/1601.07195) Unlike that project, it does not use an MPI
// library. Device parallelism is therefore ostensibly limited to multiple processors on a single node, but transparent
// OpenCL virtualization frameworks for clusters might allow this engine implementation to run as-is on a cluster, with
// appropriate cluster configuration.
//
// A single engine is broken into 2^N equal-sized subengines, for an integer "N." These engines are indexed from 0 to
// (2^N)-1 and enumerate all bit permutations from 0, at engine index 0 and substate amplitude index 0, in order up to
// the last amplitude index of engine index (2^N)-1. Refer to https://arxiv.org/abs/1601.07195 to better understand the
// single bit and controlled gate methods. When a gate operation that swaps permutation amplitudes without changing
// their magnitude or phase crosses sub-engine boundaries, it is possible to swap entire sub-engine indices instead of
// individual amplitudes, which is much cheaper.

#include <algorithm>
#include <future>

#include "oclengine.hpp"
#include "qengine_opencl_multi.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

QEngineOCLMulti::QEngineOCLMulti(
    bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, int deviceCount)
    : QInterface(qBitCount, rgp)
    , deviceIDs()
{
    clObj = OCLEngine::Instance();

    if (deviceCount < 1) {
        deviceCount = clObj->GetDeviceCount();
    }

    deviceIDs.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        deviceIDs[i] = i;
    }

    Init(qBitCount, initState);
}

QEngineOCLMulti::QEngineOCLMulti(
    bitLenInt qBitCount, bitCapInt initState, std::vector<int> devIDs, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount, rgp)
    , deviceIDs(devIDs)
{
    clObj = OCLEngine::Instance();
    Init(qBitCount, initState);
}

void QEngineOCLMulti::Init(bitLenInt qBitCount, bitCapInt initState)
{
    // It's possible to do a simple form of load balancing by assigning unequal portions of subengines to the same
    // device:
    // deviceIDs.resize(4);
    // deviceIDs[0] = 1;
    // deviceIDs[1] = 1;
    // deviceIDs[2] = 1;
    // deviceIDs[3] = 0;

    runningNorm = 1.0;

    int deviceCount = deviceIDs.size();

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

    if (deviceCount == 1) {
        substateEngines.push_back(std::make_shared<QEngineOCL>(qubitCount, initState, rand_generator, deviceIDs[0]));
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
        // All sub-engines should have zero norm except for the one containing the initialization permutation:
        substateEngines.push_back(
            std::make_shared<QEngineOCL>(subQubitCount, subInitVal, rand_generator, deviceIDs[i], partialInit));
        substateEngines[i]->EnableNormalize(false);
        subInitVal = 0;
        partialInit = true;
    }
}

// Swap the high half of one sub-engine with the low half of another. This is necessary for gates which cross sub-engine
// boundaries.
void QEngineOCLMulti::ShuffleBuffers(QEngineOCLPtr engine1, QEngineOCLPtr engine2)
{
    if ((engine1->GetCLContextID()) == (engine2->GetCLContextID())) {
        cl::Context& cntxt = engine1->GetCLContext();
        cl::Buffer tempBuffer = cl::Buffer(cntxt, CL_MEM_READ_WRITE, sizeof(complex) * (subMaxQPower >> 1));
        cl::CommandQueue& queue = engine1->GetCLQueue();
        queue.enqueueCopyBuffer(*(engine1->GetStateBuffer()), tempBuffer, sizeof(complex) * (subMaxQPower >> 1), 0, sizeof(complex) * (subMaxQPower >> 1));
        queue.enqueueCopyBuffer(*(engine2->GetStateBuffer()), *(engine1->GetStateBuffer()), 0, sizeof(complex) * (subMaxQPower >> 1), sizeof(complex) * (subMaxQPower >> 1));
        queue.enqueueCopyBuffer(tempBuffer, *(engine2->GetStateBuffer()), 0, 0, sizeof(complex) * (subMaxQPower >> 1));
        queue.finish();
    }
    else {
        engine1->LockSync(CL_MAP_READ | CL_MAP_WRITE);
        engine2->LockSync(CL_MAP_READ | CL_MAP_WRITE);
        std::swap_ranges(engine1->GetStateVector() + (subMaxQPower >> 1), engine1->GetStateVector() + subMaxQPower,
            engine2->GetStateVector());
        engine1->UnlockSync();
        engine2->UnlockSync();
    }
}

// This underlies all single-bit gates, unless they can be carried out entirely at a "meta-qubit" level, by only
// swapping sub-engine indices:
template <typename F, typename... Args>
void QEngineOCLMulti::SingleBitGate(bool doNormalize, bitLenInt bit, F fn, Args... gfnArgs)
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
    } else if (doNormalize) {
        for (i = 0; i < subEngineCount; i++) {
            substateEngines[i]->SetNorm(1.0);
            substateEngines[i]->EnableNormalize(true);
        }
    }

    if (bit < subQubitCount) {
        // In this branch, the gate is entirely independent in the parallel sub-engines
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
        // Here, the gate requires data to cross sub-engine boundaries.
        // It's always a matter of swapping the high halves of half the sub-engines with the low halves of the other
        // half of engines, acting the maximum bit gate, (for the sub-engine bit count,) and swapping back. Depending on
        // the bit index and number of sub-engines, we just have to determine which sub-engine to pair with which.
        std::vector<std::future<void>> futures(subEngineCount / 2);

        bitLenInt groupCount = 1 << (qubitCount - (bit + 1));
        bitLenInt groupSize = 1 << ((bit + 1) - subQubitCount);
        bitLenInt sqi = subQubitCount - 1;

        for (i = 0; i < groupCount; i++) {
            for (j = 0; j < (groupSize / 2); j++) {
                futures[j + (i * (groupSize / 2))] =
                    std::async(std::launch::async, [this, groupSize, i, j, fn, sqi, gfnArgs...]() {
                        QEngineOCLPtr engine1 = substateEngines[j + (i * groupSize)];
                        QEngineOCLPtr engine2 = substateEngines[j + (i * groupSize) + (groupSize / 2)];

                        ShuffleBuffers(engine1, engine2);

                        std::future<void> future1 = std::async(std::launch::async,
                            [engine1, fn, sqi, gfnArgs...]() { ((engine1.get())->*fn)(gfnArgs..., sqi); });
                        std::future<void> future2 = std::async(std::launch::async,
                            [engine2, fn, sqi, gfnArgs...]() { ((engine2.get())->*fn)(gfnArgs..., sqi); });
                        future1.get();
                        future2.get();

                        ShuffleBuffers(engine1, engine2);
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
            runningNorm += substateEngines[i]->GetNorm(false);
        }
    }

    for (i = 0; i < subEngineCount; i++) {
        substateEngines[i]->EnableNormalize(false);
    }
}

// This method underlies all controlled gates that can't be entirely carried out at the "meta-" level.
template <typename CF, typename F, typename... Args>
void QEngineOCLMulti::ControlledGate(
    bool anti, bitLenInt controlBit, bitLenInt targetBit, CF cfn, F fn, Args... gfnArgs)
{

    if (subEngineCount == 1) {
        ((substateEngines[0].get())->*cfn)(gfnArgs..., controlBit, targetBit);
        return;
    }

    if ((controlBit >= subQubitCount) && (targetBit >= subQubitCount)) {
        // If both the control and target are "meta-," we can do this at a "meta-" level with a single-bit gate
        // "payload."
        MetaControlled(anti, { static_cast<bitLenInt>(controlBit - subQubitCount) },
            static_cast<bitLenInt>(targetBit - subQubitCount), fn, gfnArgs...);
    } else if (controlBit >= subQubitCount) {
        // If the control is "meta-," but the target is not, we do this semi- at a "meta-" level, again with a
        // single-bit gate payload.
        SemiMetaControlled(anti, { static_cast<bitLenInt>(controlBit - subQubitCount) }, targetBit, fn, gfnArgs...);
    } else if (controlBit < (subQubitCount - 1)) {
        // If both control and target bits fit within independent sub-engines, we can just do this parallelized over
        // sub-engines with a two-bit gate.
        SingleBitGate(false, targetBit, cfn, gfnArgs..., controlBit);
    } else {
        // There's a particular edge case not handled by any of the above, where both control and target fit in
        // independent sub-engines, but the control bit is the highest bit in the sub-engine.
        ControlledSkip(anti, 0, targetBit, fn, gfnArgs...);
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

    // Like singly-controlled gates, we can handle this entirely "meta-," "semi-meta-," or parallelized entirely
    // independently within sub-engines.
    if ((lowControl >= subQubitCount) && (targetBit >= subQubitCount)) {
        MetaControlled(anti,
            { static_cast<bitLenInt>(controlBit1 - subQubitCount),
                static_cast<bitLenInt>(controlBit2 - subQubitCount) },
            static_cast<bitLenInt>(targetBit - subQubitCount), fn, gfnArgs...);
    } else if (lowControl < (subQubitCount - 1)) {
        ControlledGate(anti, highControl, targetBit, ccfn, cfn, gfnArgs..., lowControl);
    } else if (lowControl >= subQubitCount) {
        // Both controls >= subQubitCount, targetBit < subQubitCount
        SemiMetaControlled(anti,
            { static_cast<bitLenInt>(lowControl - subQubitCount), static_cast<bitLenInt>(highControl - subQubitCount) },
            targetBit, fn, gfnArgs...);
    } else if ((highControl >= subQubitCount) && (lowControl != (subQubitCount - 1))) {
        if (targetBit >= subQubitCount) {
            MetaControlled(anti, { static_cast<bitLenInt>(highControl - subQubitCount) },
                static_cast<bitLenInt>(targetBit - subQubitCount), cfn, gfnArgs..., lowControl);
        } else {
            SemiMetaControlled(
                anti, { static_cast<bitLenInt>(highControl - subQubitCount) }, targetBit, cfn, gfnArgs..., lowControl);
        }
    } else {
        // Again, there's a particular edge case not handled by any of the above, where the low control bit and the
        // target bit fit in independent sub-engines, but the low control bit is the highest bit in the sub-engine.
        ControlledSkip(anti, 1, targetBit, fn, gfnArgs...);
    }
}

// This method has not been parallelized.
void QEngineOCLMulti::SetQuantumState(complex* inputState)
{
    CombineEngines(qubitCount - 1);
    substateEngines[0]->SetQuantumState(inputState);
    SeparateEngines();
}

void QEngineOCLMulti::SetPermutation(bitCapInt perm)
{
    if (subEngineCount == 1) {
        substateEngines[0]->SetPermutation(perm);
        return;
    }

    std::vector<std::future<void>> futures(subEngineCount);
    bitCapInt i;
    bitCapInt j = 0;
    for (i = 0; i < maxQPower; i += subMaxQPower) {
        if ((perm >= i) && (perm < (i + subMaxQPower))) {
            QEngineOCLPtr engine = substateEngines[j];
            bitCapInt p = perm - i;
            futures[j] = std::async(std::launch::async, [engine, p]() { engine->SetPermutation(p); });
        } else {
            futures[j] = std::async(std::launch::async, [this, j]() { substateEngines[j]->NormalizeState(0.0); });
        }
        j++;
    }
    for (i = 0; i < subEngineCount; i++) {
        futures[i].get();
    }
}

// Certain difficult and/or less essential methods have not been parallelized.
bitLenInt QEngineOCLMulti::Cohere(QEngineOCLMultiPtr toCopy)
{
    bitLenInt result;
    CombineEngines(qubitCount - 1);
    toCopy->CombineEngines(toCopy->GetQubitCount() - 1);
    result = substateEngines[0]->Cohere(toCopy->substateEngines[0]);
    SetQubitCount(qubitCount + toCopy->qubitCount);
    SeparateEngines();
    toCopy->SeparateEngines();
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
    CombineEngines(qubitCount - 1);
    dest->CombineEngines(dest->GetQubitCount() - 1);
    substateEngines[0]->Decohere(start, length, dest->substateEngines[0]);
    if (qubitCount <= length) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }
    SeparateEngines();
    dest->SeparateEngines();
}

void QEngineOCLMulti::Dispose(bitLenInt start, bitLenInt length)
{
    CombineEngines(qubitCount - 1);
    substateEngines[0]->Dispose(start, length);
    if (qubitCount <= length) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }
    SeparateEngines();
}

// Single-bit gate wrappers and length variants follow. Some can have certain cases offloaded to (cheaper) "meta-"
// variants.
void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if ((control1 >= subQubitCount) && (control2 >= subQubitCount) && (target >= subQubitCount)) {

        control1 -= subQubitCount;
        control2 -= subQubitCount;
        target -= subQubitCount;

        MetaCNOT(false, { control1, control2 }, target);
    } else {
        DoublyControlledGate(false, control1, control2, target, (CCGFn)(&QEngineOCL::CCNOT), (CGFn)(&QEngineOCL::CNOT),
            (GFn)(&QEngineOCL::X));
    }
}

void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target)
{
    if ((control >= subQubitCount) && (target >= subQubitCount)) {

        control -= subQubitCount;
        target -= subQubitCount;

        MetaCNOT(false, { control }, target);
    } else {
        ControlledGate(false, control, target, (CGFn)(&QEngineOCL::CNOT), (GFn)(&QEngineOCL::X));
    }
}

void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if ((control1 >= subQubitCount) && (control2 >= subQubitCount) && (target >= subQubitCount)) {

        control1 -= subQubitCount;
        control2 -= subQubitCount;
        target -= subQubitCount;

        MetaCNOT(true, { control1, control2 }, target);
    } else {
        DoublyControlledGate(true, control1, control2, target, (CCGFn)(&QEngineOCL::AntiCCNOT),
            (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
    }
}

void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target)
{
    if ((control >= subQubitCount) && (target >= subQubitCount)) {

        control -= subQubitCount;
        target -= subQubitCount;

        MetaCNOT(true, { control }, target);
    } else {
        ControlledGate(true, control, target, (CGFn)(&QEngineOCL::AntiCNOT), (GFn)(&QEngineOCL::X));
    }
}

void QEngineOCLMulti::H(bitLenInt qubitIndex) { SingleBitGate(true, qubitIndex, (GFn)(&QEngineOCL::H)); }

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
        std::vector<std::future<void>> futures(subEngineCount / 2);
        bitLenInt groupCount = 1 << (qubitCount - (qubit + 1));
        bitLenInt groupSize = 1 << ((qubit + 1) - subQubitCount);
        bitLenInt keepOffset, clearOffset;
        if (result) {
            keepOffset = 1;
            clearOffset = 0;
        } else {
            keepOffset = 0;
            clearOffset = 1;
        }
        for (i = 0; i < groupCount; i++) {
            for (j = 0; j < (groupSize / 2); j++) {
                futures[j + (i * (groupSize / 2))] =
                    std::async(std::launch::async, [this, i, j, &groupSize, &clearOffset, &keepOffset, &nrmlzr]() {
                        bitLenInt clearIndex = j + (i * groupSize) + (clearOffset * groupSize / 2);
                        bitLenInt keepIndex = j + (i * groupSize) + (keepOffset * groupSize / 2);

                        substateEngines[clearIndex]->NormalizeState(0.0);
                        substateEngines[keepIndex]->NormalizeState(nrmlzr);

                    });
            }
        }

        for (i = 0; i < (subEngineCount / 2); i++) {
            futures[i].get();
        }
    }

    return result;
}

// See QEngineCPU::X(start, length) in src/qengine/state/gates.cpp
void QEngineOCLMulti::MetaX(bitLenInt start, bitLenInt length)
{
    bitCapInt targetMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = (subEngineCount - 1) ^ targetMask;

    std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount);

    par_for(0, subEngineCount, [&](const bitCapInt lcv, const int cpu) {
        nSubstateEngines[(lcv & otherMask) | (lcv ^ targetMask)] = substateEngines[lcv];
    });

    for (bitLenInt i = 0; i < subEngineCount; i++) {
        substateEngines[i] = nSubstateEngines[i];
    }
    SetQubitCount(qubitCount);
}

void QEngineOCLMulti::MetaCNOT(bool anti, std::vector<bitLenInt> controls, bitLenInt target)
{
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controls.size(); i++) {
        controlMask |= 1 << controls[i];
    }
    bitCapInt testMask = anti ? 0 : controlMask;

    bitCapInt targetMask = 1 << target;
    bitCapInt otherMask = (subEngineCount - 1) ^ (controlMask | targetMask);

    std::vector<QEngineOCLPtr> nSubstateEngines(subEngineCount);

    par_for(0, subEngineCount, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & otherMask);
        bitCapInt controlRes = (lcv & controlMask);
        if ((lcv & controlMask) == testMask) {
            nSubstateEngines[controlRes | otherRes | ((~lcv) & targetMask)] = substateEngines[lcv];
        } else {
            nSubstateEngines[lcv] = substateEngines[lcv];
        }
    });

    for (bitLenInt i = 0; i < subEngineCount; i++) {
        substateEngines[i] = nSubstateEngines[i];
    }
    SetQubitCount(qubitCount);
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename F, typename... Args>
void QEngineOCLMulti::MetaControlled(
    bool anti, std::vector<bitLenInt> controls, bitLenInt target, F fn, Args... gfnArgs)
{
    bitLenInt i;

    std::vector<bitLenInt> sortedMasks(1 + controls.size());
    sortedMasks[controls.size()] = 1 << target;

    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1 << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }

    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt targetMask = 1 << target;
    bitLenInt sqi = subQubitCount - 1;

    bitLenInt maxLCV = subEngineCount >> (sortedMasks.size());
    std::vector<std::future<void>> futures(maxLCV);

    for (i = 0; i < maxLCV; i++) {
        futures[i] =
            std::async(std::launch::async, [this, i, &sortedMasks, &controlMask, &targetMask, &sqi, fn, gfnArgs...]() {

                bitCapInt j, k, jLo, jHi;
                jHi = i;
                j = 0;
                for (k = 0; k < (sortedMasks.size()); k++) {
                    jLo = jHi & sortedMasks[k];
                    jHi = (jHi ^ jLo) << 1;
                    j |= jLo;
                }
                j |= jHi | controlMask;

                QEngineOCLPtr engine1 = substateEngines[j];
                QEngineOCLPtr engine2 = substateEngines[j + targetMask];

                ShuffleBuffers(engine1, engine2);

                std::future<void> future1 = std::async(
                    std::launch::async, [engine1, fn, sqi, gfnArgs...]() { ((engine1.get())->*fn)(gfnArgs..., sqi); });
                std::future<void> future2 = std::async(
                    std::launch::async, [engine2, fn, sqi, gfnArgs...]() { ((engine2.get())->*fn)(gfnArgs..., sqi); });
                future1.get();
                future2.get();

                ShuffleBuffers(engine1, engine2);
            });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to fit in
// sub-engines).
template <typename F, typename... Args>
void QEngineOCLMulti::SemiMetaControlled(
    bool anti, std::vector<bitLenInt> controls, bitLenInt targetBit, F fn, Args... gfnArgs)
{
    bitLenInt i;
    bitLenInt maxLCV = subEngineCount >> (controls.size());
    std::vector<bitLenInt> sortedMasks(controls.size());
    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1 << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    std::vector<std::future<void>> futures(maxLCV);
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, fn, sortedMasks, controlMask, targetBit, gfnArgs...]() {
            bitCapInt j, k, jLo, jHi;
            jHi = i;
            j = 0;
            for (k = 0; k < (sortedMasks.size()); k++) {
                jLo = jHi & sortedMasks[k];
                jHi = (jHi ^ jLo) << 1;
                j |= jLo;
            }
            j |= jHi | controlMask;

            (substateEngines[j].get()->*fn)(gfnArgs..., targetBit);
        });
    }
    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is the particular edge case between any "meta-" gate and any gate entirely below the "meta-"/"embarrassingly
// parallel" threshold, where a control bit is the most significant bit in a sub-engine.
template <typename F, typename... Args>
void QEngineOCLMulti::ControlledSkip(bool anti, bitLenInt controlDepth, bitLenInt targetBit, F fn, Args... gfnArgs)
{
    bitLenInt i, j;
    bitLenInt k = 0;
    bitLenInt groupCount = 1 << (qubitCount - (targetBit + 1));
    bitLenInt groupSize = 1 << ((targetBit + 1) - subQubitCount);
    std::vector<std::future<void>> futures((groupCount * groupSize) / 2);
    bitLenInt sqi = subQubitCount - 1;
    bitLenInt jStart = (anti | (controlDepth == 0)) ? 0 : ((groupSize / 2) - 1);
    bitLenInt jInc = (controlDepth == 0) ? 1 : 2;

    for (i = 0; i < groupCount; i++) {
        for (j = jStart; j < (groupSize / 2); j += jInc) {
            futures[k] = std::async(std::launch::async, [this, groupSize, i, j, fn, sqi, anti, gfnArgs...]() {
                QEngineOCLPtr engine1 = substateEngines[j + (i * groupSize)];
                QEngineOCLPtr engine2 = substateEngines[j + (i * groupSize) + (groupSize / 2)];

                ShuffleBuffers(engine1, engine2);

                if (anti) {
                    ((engine1.get())->*fn)(gfnArgs..., sqi);
                } else {
                    ((engine2.get())->*fn)(gfnArgs..., sqi);
                }

                ShuffleBuffers(engine1, engine2);
            });
            k++;
        }
    }
    for (i = 0; i < k; i++) {
        futures[i].get();
    }
}

void QEngineOCLMulti::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex)
{
    SingleBitGate(doCalcNorm, qubitIndex, (ASBFn)(&QEngineOCL::ApplySingleBit), mtrx, doCalcNorm);
}

void QEngineOCLMulti::X(bitLenInt qubitIndex)
{
    if (qubitIndex >= subQubitCount) {
        MetaX(qubitIndex - subQubitCount, 1);
    } else {
        SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::X));
    }
}

void QEngineOCLMulti::Y(bitLenInt qubitIndex) { SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Y)); }

void QEngineOCLMulti::Z(bitLenInt qubitIndex) { SingleBitGate(false, qubitIndex, (GFn)(&QEngineOCL::Z)); }

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
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RT), radians);
}
void QEngineOCLMulti::RX(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::RY(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::RZ(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::Exp(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::Exp), radians);
}
void QEngineOCLMulti::ExpX(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::ExpX), radians);
}
void QEngineOCLMulti::ExpY(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::ExpY), radians);
}
void QEngineOCLMulti::ExpZ(real1 radians, bitLenInt qubitIndex)
{
    SingleBitGate(true, qubitIndex, (RGFn)(&QEngineOCL::ExpZ), radians);
}
void QEngineOCLMulti::CRX(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRX), (RGFn)(&QEngineOCL::RX), radians);
}
void QEngineOCLMulti::CRY(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRY), (RGFn)(&QEngineOCL::RY), radians);
}
void QEngineOCLMulti::CRZ(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRZ), (RGFn)(&QEngineOCL::RZ), radians);
}
void QEngineOCLMulti::CRT(real1 radians, bitLenInt control, bitLenInt target)
{
    ControlledGate(false, control, target, (CRGFn)(&QEngineOCL::CRT), (RGFn)(&QEngineOCL::RT), radians);
}

// Non-primitive gates combine the bare minimum number of sub-engines into a single engine, in order to call the
// underlying non-primitive gate on the engine.
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
void QEngineOCLMulti::MUL(
    bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bool clearCarry)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->MUL(toMul, inOutStart, carryStart, length, clearCarry); },
        { static_cast<bitLenInt>(inOutStart + length - 1), static_cast<bitLenInt>(carryStart + length - 1) });
}
void QEngineOCLMulti::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->DIV(toDiv, inOutStart, carryStart, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1), static_cast<bitLenInt>(carryStart + length - 1) });
}
void QEngineOCLMulti::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit,
    bitLenInt length, bool clearCarry)
{
    CombineAndOp(
        [&](QEngineOCLPtr engine) { engine->CMUL(toMul, inOutStart, carryStart, controlBit, length, clearCarry); },
        { static_cast<bitLenInt>(inOutStart + length - 1), static_cast<bitLenInt>(carryStart + length - 1),
            controlBit });
}
void QEngineOCLMulti::CDIV(
    bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt controlBit, bitLenInt length)
{
    CombineAndOp([&](QEngineOCLPtr engine) { engine->CDIV(toDiv, inOutStart, carryStart, controlBit, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1), static_cast<bitLenInt>(carryStart + length - 1),
            controlBit });
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
    if ((start + length) > subQubitCount) {
        bitLenInt s, len;
        if (start > subQubitCount) {
            s = start - subQubitCount;
            len = length;
            length = 0;
        } else {
            s = 0;
            len = (start + length) - subQubitCount;
            length -= len;
        }

        MetaX(s, len);
    }
    if (length > 0) {
        RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->X(start, len); },
            [&](bitLenInt offset) { X(start + offset); }, length, { static_cast<bitLenInt>(start + length - 1) });
    }
}

bitLenInt QEngineOCLMulti::SeparateMetaCNOT(
    bool anti, std::vector<bitLenInt> controls, bitLenInt target, bitLenInt length)
{
    bitLenInt i, j;
    bitLenInt lowStart = qubitCount;
    for (i = 0; i < controls.size(); i++) {
        if (controls[i] < lowStart) {
            lowStart = controls[i];
        }
    }
    if (target < lowStart) {
        lowStart = target;
    }

    if ((lowStart + length) <= subQubitCount) {
        return length;
    }

    bitLenInt len;
    if (lowStart >= subQubitCount) {
        for (i = 0; i < controls.size(); i++) {
            controls[i] -= subQubitCount;
        }
        target -= subQubitCount;
        len = length;
        length = 0;
    } else {
        for (i = 0; i < controls.size(); i++) {
            controls[i] -= lowStart;
        }
        target -= lowStart;
        len = (lowStart + length) - subQubitCount;
        length -= len;
    }

    for (i = 0; i < len; i++) {
        MetaCNOT(anti, controls, target + i);
        for (j = 0; j < controls.size(); j++) {
            controls[j]++;
        }
    }

    return length;
}

void QEngineOCLMulti::CNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    length = SeparateMetaCNOT(false, { control }, target, length);
    if (length > 0) {
        RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->CNOT(control, target, len); },
            [&](bitLenInt offset) { CNOT(control + offset, target + offset); }, length,
            { static_cast<bitLenInt>(control + length - 1), static_cast<bitLenInt>(target + length - 1) });
    }
}

void QEngineOCLMulti::AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length)
{
    length = SeparateMetaCNOT(true, { control }, target, length);
    if (length > 0) {
        RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->AntiCNOT(control, target, len); },
            [&](bitLenInt offset) { AntiCNOT(control + offset, target + offset); }, length,
            { static_cast<bitLenInt>(control + length - 1), static_cast<bitLenInt>(target + length - 1) });
    }
}

void QEngineOCLMulti::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    length = SeparateMetaCNOT(false, { control1, control2 }, target, length);
    if (length > 0) {
        RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->CCNOT(control1, control2, target, len); },
            [&](bitLenInt offset) { CCNOT(control1 + offset, control2 + offset, target + offset); }, length,
            { static_cast<bitLenInt>(control1 + length - 1), static_cast<bitLenInt>(control2 + length - 1),
                static_cast<bitLenInt>(target + length - 1) });
    }
}

void QEngineOCLMulti::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
{
    length = SeparateMetaCNOT(false, { control1, control2 }, target, length);
    if (length > 0) {
        RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->AntiCCNOT(control1, control2, target, len); },
            [&](bitLenInt offset) { AntiCCNOT(control1 + offset, control2 + offset, target + offset); }, length,
            { static_cast<bitLenInt>(control1 + length - 1), static_cast<bitLenInt>(control2 + length - 1),
                static_cast<bitLenInt>(target + length - 1) });
    }
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
    CombineAndOp(
        [&](QEngineOCLPtr engine) { engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values); },
        { static_cast<bitLenInt>(indexStart + indexLength - 1), static_cast<bitLenInt>(valueStart + valueLength - 1) });

    return 0;
}

bitCapInt QEngineOCLMulti::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOp(
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
    CombineAndOp(
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
        CNOT(qubitIndex1, qubitIndex2);
        CNOT(qubitIndex2, qubitIndex1);
        CNOT(qubitIndex1, qubitIndex2);
    }
}

void QEngineOCLMulti::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    RegOp([&](QEngineOCLPtr engine, bitLenInt len) { engine->Swap(start1, start2, len); },
        [&](bitLenInt offset) { Swap(start1 + offset, start2 + offset); }, length,
        { static_cast<bitLenInt>(start1 + length - 1), static_cast<bitLenInt>(start2 + length - 1) });
}

void QEngineOCLMulti::CopyState(QEngineOCLMultiPtr orig)
{
    CombineEngines(qubitCount - 1);
    orig->CombineEngines(orig->GetQubitCount() - 1);
    substateEngines[0]->CopyState(orig->substateEngines[0]);
    SeparateEngines();
    orig->SeparateEngines();
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
void QEngineOCLMulti::CombineEngines(bitLenInt bit)
{

    if (subEngineCount == 1) {
        return;
    }

    bitLenInt i, j;

    bitLenInt order = (qubitCount - (bit + 1));
    bitLenInt groupCount = 1 << order;
    bitLenInt groupSize = 1 << ((bit + 1) - subQubitCount);
    std::vector<QEngineOCLPtr> nEngines(groupCount);
    bitCapInt sbSize = maxQPower / subEngineCount;

    bool isMapped;

    for (i = 0; i < groupCount; i++) {
        nEngines[i] = std::make_shared<QEngineOCL>((bit + 1), 0, rand_generator, deviceIDs[i]);
        nEngines[i]->EnableNormalize(false);
        isMapped = false;
        complex* nsv = nullptr;
        for (j = 0; j < groupSize; j++) {
            QEngineOCLPtr eng = substateEngines[j + (i * groupSize)];
            if ((nEngines[i]->GetCLContextID()) == (eng->GetCLContextID())) {
                cl::CommandQueue& queue = eng->GetCLQueue();
                queue.enqueueCopyBuffer(*(eng->GetStateBuffer()), *(nEngines[i]->GetStateBuffer()), 0, j * sizeof(complex) * sbSize, sizeof(complex) * sbSize);
                queue.finish();
            } else {
                if (!isMapped) {
                    nEngines[i]->LockSync(CL_MAP_WRITE);
                    nsv = nEngines[i]->GetStateVector();
                    isMapped = true;
                }
                complex* sv = eng->GetStateVector();
                eng->LockSync(CL_MAP_READ);
                std::copy(sv, sv + sbSize, nsv + (j * sbSize));
                eng->UnlockSync();
            }
        }
        if (isMapped) {
            nEngines[i]->UnlockSync();
        }
    }

    if (order == 0) {
        nEngines[0]->EnableNormalize(true);
    }

    substateEngines.resize(groupCount);
    for (i = 0; i < groupCount; i++) {
        substateEngines[i] = nEngines[i];
    }
    SetQubitCount(qubitCount);
}

void QEngineOCLMulti::SeparateEngines()
{

    bitLenInt engineCount = 1 << maxDeviceOrder;

    if (maxDeviceOrder >= qubitCount) {
        engineCount = 1 << (qubitCount - 1);
    }

    if (engineCount <= subEngineCount) {
        return;
    }

    bitLenInt i, j;

    bitLenInt groupSize = engineCount / subEngineCount;

    std::vector<QEngineOCLPtr> nEngines(engineCount);
    bitCapInt sbSize = maxQPower / engineCount;

    bool isMapped;

    for (i = 0; i < subEngineCount; i++) {
        QEngineOCLPtr eng = substateEngines[i];
        isMapped = false;
        complex* sv = nullptr;
        for (j = 0; j < groupSize; j++) {
            QEngineOCLPtr nEngine =
                std::make_shared<QEngineOCL>(qubitCount - log2(engineCount), 0, rand_generator, deviceIDs[j], true);
            nEngine->EnableNormalize(false);
            if ((nEngine->GetCLContextID()) == (eng->GetCLContextID())) {
                cl::CommandQueue& queue = eng->GetCLQueue();
                queue.enqueueCopyBuffer(*(eng->GetStateBuffer()), *(nEngine->GetStateBuffer()), j * sizeof(complex) * sbSize, 0, sizeof(complex) * sbSize);
                queue.finish();
            } else {
                if (!isMapped) {
                    eng->LockSync(CL_MAP_READ);
                    sv = eng->GetStateVector();
                    isMapped = true;
                }
                nEngine->LockSync(CL_MAP_WRITE);
                std::copy(sv + (j * sbSize), sv + ((j + 1) * sbSize), nEngine->GetStateVector());
                nEngine->UnlockSync();
            }
            nEngines[j + (i * groupSize)] = nEngine;
        }
        if (isMapped) {
            eng->UnlockSync();
        }
    }

    substateEngines.resize(engineCount);
    for (i = 0; i < engineCount; i++) {
        substateEngines[i] = nEngines[i];
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

    if (highestBit >= subQubitCount) {
        CombineEngines(highestBit);
    }

    std::vector<std::future<void>> futures(subEngineCount);
    for (i = 0; i < subEngineCount; i++) {
        futures[i] = std::async(std::launch::async, [this, fn, i]() { fn(substateEngines[i]); });
    }
    for (i = 0; i < subEngineCount; i++) {
        futures[i].get();
    }

    if (highestBit >= subQubitCount) {
        SeparateEngines();
    }
}

template <typename F, typename OF>
void QEngineOCLMulti::RegOp(F fn, OF ofn, bitLenInt length, std::vector<bitLenInt> bits)
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
        int subLength = ((int)length) - ((int)bitDiff);
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

void QEngineOCLMulti::NormalizeState(real1 nrm)
{
    if (nrm < 0.0) {
        nrm = runningNorm;
    }
    if ((nrm <= 0.0) || (nrm == 1.0)) {
        return;
    }

    bitLenInt i;
    if (runningNorm != 1.0) {
        std::vector<std::future<void>> nf(subEngineCount);
        for (i = 0; i < subEngineCount; i++) {
            nf[i] = std::async(std::launch::async, [this, i]() { substateEngines[i]->NormalizeState(runningNorm); });
        }
        for (i = 0; i < subEngineCount; i++) {
            nf[i].get();
        }
    }
    runningNorm = 1.0;
}

} // namespace Qrack
