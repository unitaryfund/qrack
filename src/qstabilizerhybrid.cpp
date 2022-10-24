//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#include <thread>

#define IS_REAL_1(r) (abs(ONE_CMPLX - r) <= FP_NORM_EPSILON)
#define IS_CTRLED_CLIFFORD(top, bottom)                                                                                \
    ((IS_REAL_1(std::real(top)) || IS_REAL_1(std::imag(bottom))) && (IS_SAME(top, bottom) || IS_SAME(top, -bottom)))
#define IS_CLIFFORD_PHASE_INVERT(top, bottom)                                                                          \
    (IS_SAME(top, bottom) || IS_SAME(top, -bottom) || IS_SAME(top, I_CMPLX * bottom) || IS_SAME(top, -I_CMPLX * bottom))
#define IS_CLIFFORD(mtrx)                                                                                              \
    ((IS_PHASE(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[0], mtrx[3])) ||                                                 \
        (IS_INVERT(mtrx) && IS_CLIFFORD_PHASE_INVERT(mtrx[1], mtrx[2])) ||                                             \
        ((IS_SAME(mtrx[0U], mtrx[1U]) || IS_SAME(mtrx[0U], -mtrx[1U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) ||      \
             IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                                                                \
            (IS_SAME(mtrx[0U], mtrx[2U]) || IS_SAME(mtrx[0U], -mtrx[2U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                                                             \
            (IS_SAME(mtrx[0U], mtrx[3U]) || IS_SAME(mtrx[0U], -mtrx[3U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) ||   \
                IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U]))))
#define IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , useHostRam(useHostMem)
    , doNormalize(doNorm)
    , isSparse(useSparseStateVec)
    , useTGadget(true)
    , thresholdQubits(qubitThreshold)
    , ancillaCount(0)
    , maxQubitPlusAncillaCount(28)
    , separabilityThreshold(sep_thresh)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , engine(NULL)
    , deviceIDs(devList)
    , engineTypes(eng)
    , cloneEngineTypes(eng)
    , shards(qubitCount)
{

#if ENABLE_OPENCL
    DeviceContextPtr devContext = OCLEngine::Instance().GetDeviceContextPtr(devID);
    const bitLenInt maxPageQubits = log2(devContext->GetMaxAlloc() / sizeof(complex));
#else
    const bitLenInt maxPageQubits =
        getenv("QRACK_MAX_CPU_QB") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_CPU_QB"))) - 2U : 30U;
#endif

#if ENABLE_ENV_VARS
    maxQubitPlusAncillaCount = getenv("QRACK_MAX_PAGING_QB")
        ? (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")))
        : maxPageQubits + 2U;
#else
    maxQubitPlusAncillaCount = maxPageQubits + 2U;
#endif

    stabilizer = MakeStabilizer(initState);
}

QStabilizerPtr QStabilizerHybrid::MakeStabilizer(bitCapInt perm)
{
    return std::make_shared<QStabilizer>(qubitCount + ancillaCount, perm, rand_generator, CMPLX_DEFAULT_ARG, false,
        randGlobalPhase, false, -1, useRDRAND);
}
QInterfacePtr QStabilizerHybrid::MakeEngine(bitCapInt perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineTypes, qubitCount, perm, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
        thresholdQubits, separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return toRet;
}
QInterfacePtr QStabilizerHybrid::MakeEngine(bitCapInt perm, bitLenInt qbCount)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineTypes, qbCount, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return toRet;
}

void QStabilizerHybrid::InvertBuffer(bitLenInt qubit)
{
    complex pauliX[4U]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
    pauliShard->Compose(shards[qubit]->gate);
    shards[qubit] = pauliShard->IsIdentity() ? NULL : pauliShard;
    stabilizer->X(qubit);
}

void QStabilizerHybrid::FlushH(bitLenInt qubit)
{
    complex hGate[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
        -complex(SQRT1_2_R1, ZERO_R1) };
    MpsShardPtr shard = std::make_shared<MpsShard>(hGate);
    shard->Compose(shards[qubit]->gate);
    shards[qubit] = shard->IsIdentity() ? NULL : shard;
    stabilizer->H(qubit);
}

void QStabilizerHybrid::FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase)
{
    if (engine) {
        return;
    }

    MpsShardPtr shard = shards[control];
    if (shard && (shard->IsHPhase() || shard->IsHInvert())) {
        FlushH(control);
    }
    shard = shards[control];
    if (shard && shard->IsInvert()) {
        InvertBuffer(control);
    }
    shard = shards[control];
    if (shard && !shard->IsPhase()) {
        SwitchToEngine();
        return;
    }

    shard = shards[target];
    if (shard && (shard->IsHPhase() || shard->IsHInvert())) {
        FlushH(target);
    }
    shard = shards[target];
    if (shard && shard->IsInvert()) {
        InvertBuffer(target);
    }

    shard = shards[target];
    if (!shard) {
        return;
    }
    // Shard is definitely non-NULL.

    if (!(shard->IsPhase())) {
        SwitchToEngine();
        return;
    }
    // Shard is definitely a phase gate.

    if (isPhase) {
        return;
    }
    // The gate payload is definitely not a phase gate.
    // This is the new case we can handle with the "reverse gadget" for t-injection in this PRX Quantum article, in
    // Appendix A: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020361
    // Hakop Pashayan, Oliver Reardon-Smith, Kamil Korzekwa, and Stephen D. Bartlett
    // PRX Quantum 3, 020361 – Published 23 June 2022

    if (!useTGadget || ((qubitCount + ancillaCount) >= maxQubitPlusAncillaCount)) {
        // The option to optimize this case is off.
        SwitchToEngine();
        return;
    }

    shards[target] = NULL;

    QStabilizerPtr ancilla = std::make_shared<QStabilizer>(
        1U, 0U, rand_generator, CMPLX_DEFAULT_ARG, false, randGlobalPhase, false, -1, useRDRAND);

    // Form potentially entangled representation, with this.
    bitLenInt ancillaIndex = stabilizer->Compose(ancilla);
    ++ancillaCount;
    shards.push_back(NULL);

    // Use reverse t-injection gadget.
    stabilizer->CNOT(target, ancillaIndex);
    Mtrx(shard->gate, ancillaIndex);
    H(ancillaIndex);

    // When we measure, we act postselection, but not yet.
    // ForceM(ancillaIndex, false, true, true);
    // Ancilla is separable after measurement.
    // Dispose(ancillaIndex, 1U);
}

bool QStabilizerHybrid::CollapseSeparableShard(bitLenInt qubit)
{
    MpsShardPtr shard = shards[qubit];
    shards[qubit] = NULL;

    const bool isZ1 = stabilizer->M(qubit);
    const real1_f prob = (real1_f)((isZ1) ? norm(shard->gate[3U]) : norm(shard->gate[2U]));

    bool result;
    if (prob <= ZERO_R1) {
        result = false;
    } else if (prob >= ONE_R1) {
        result = true;
    } else {
        result = (Rand() <= prob);
    }

    if (result != isZ1) {
        stabilizer->X(qubit);
    }

    return result;
}

void QStabilizerHybrid::FlushBuffers()
{
    if (stabilizer) {
        if (IsBuffered()) {
            // This will call FlushBuffers() again after no longer stabilizer.
            SwitchToEngine();
        }
        return;
    }

    for (size_t i = 0U; i < shards.size(); ++i) {
        MpsShardPtr shard = shards[i];
        if (shard) {
            shards[i] = NULL;
            engine->Mtrx(shard->gate, i);
        }
    }
}

bool QStabilizerHybrid::TrimControls(const std::vector<bitLenInt>& lControls, std::vector<bitLenInt>& output, bool anti)
{
    if (engine) {
        output.insert(output.begin(), lControls.begin(), lControls.end());
        return false;
    }

    for (size_t i = 0U; i < lControls.size(); ++i) {
        bitLenInt bit = lControls[i];

        if (!stabilizer->IsSeparableZ(bit)) {
            output.push_back(bit);
            continue;
        }

        if (shards[bit]) {
            if (shards[bit]->IsInvert()) {
                if (anti != stabilizer->M(bit)) {
                    return true;
                }
                continue;
            }

            if (shards[bit]->IsPhase()) {
                if (anti == stabilizer->M(bit)) {
                    return true;
                }
                continue;
            }

            output.push_back(bit);
        } else if (anti == stabilizer->M(bit)) {
            return true;
        }
    }

    return false;
}

void QStabilizerHybrid::CacheEigenstate(bitLenInt target)
{
    if (engine) {
        return;
    }

    MpsShardPtr toRet = NULL;
    // If in PauliX or PauliY basis, compose gate with conversion from/to PauliZ basis.
    stabilizer->H(target);
    if (stabilizer->IsSeparableZ(target)) {
        // X eigenstate
        const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(SQRT1_2_R1, ZERO_R1), complex(-SQRT1_2_R1, ZERO_R1) };
        toRet = std::make_shared<MpsShard>(mtrx);
    } else {
        stabilizer->H(target);
        stabilizer->IS(target);
        stabilizer->H(target);
        if (stabilizer->IsSeparableZ(target)) {
            // Y eigenstate
            const complex mtrx[4U]{ complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
                complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };
            toRet = std::make_shared<MpsShard>(mtrx);
        } else {
            stabilizer->H(target);
            stabilizer->S(target);
        }
    }

    if (!toRet) {
        return;
    }

    if (shards[target]) {
        toRet->Compose(shards[target]->gate);
    }

    shards[target] = toRet;
}

QInterfacePtr QStabilizerHybrid::Clone()
{
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(cloneEngineTypes, qubitCount, 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);

    if (engine) {
        // Clone and set engine directly.
        c->engine = engine->Clone();
        c->stabilizer = NULL;
        return c;
    }

    // Otherwise, stabilizer
    c->engine = NULL;
    c->stabilizer = std::dynamic_pointer_cast<QStabilizer>(stabilizer->Clone());
    c->shards.resize(shards.size());
    c->ancillaCount = ancillaCount;
    for (size_t i = 0U; i < shards.size(); ++i) {
        if (shards[i]) {
            c->shards[i] = std::make_shared<MpsShard>(shards[i]->gate);
        }
    }

    return c;
}

void QStabilizerHybrid::SwitchToEngine()
{
    if (engine) {
        return;
    }

    engine = MakeEngine(0, stabilizer->GetQubitCount());
    if (engineTypes.size() && (engineTypes[0] == QINTERFACE_BDT)) {
        std::dynamic_pointer_cast<QBdt>(engine)->SetStateVector();
    }
    stabilizer->GetQuantumState(engine);
    if (engineTypes.size() && (engineTypes[0] == QINTERFACE_BDT)) {
        std::dynamic_pointer_cast<QBdt>(engine)->ResetStateVector();
    }
    stabilizer = NULL;
    FlushBuffers();

    if (!ancillaCount) {
        return;
    }

    // When we measure, we act postselection on reverse T-gadgets.
    engine->ForceMReg(qubitCount, ancillaCount, 0, true, true);
    // Ancillae are separable after measurement.
    engine->Dispose(qubitCount, ancillaCount);
    // We have extra "gate fusion" shards leftover.
    shards.erase(shards.begin() + qubitCount, shards.end());
    // We're done with ancillae.
    ancillaCount = 0;
}

bitLenInt QStabilizerHybrid::ComposeEither(QStabilizerHybridPtr toCopy, bool willDestroy)
{
    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    const bitLenInt nQubits = qubitCount + toCopy->qubitCount;

    if ((nQubits <= maxQubitPlusAncillaCount) &&
        (nQubits + ancillaCount + toCopy->ancillaCount) > maxQubitPlusAncillaCount) {
        SwitchToEngine();
    }

    bitLenInt toRet;
    if (engine) {
        toCopy->SwitchToEngine();
        toRet = willDestroy ? engine->ComposeNoClone(toCopy->engine) : engine->Compose(toCopy->engine);
    } else if (toCopy->engine) {
        SwitchToEngine();
        toRet = willDestroy ? engine->ComposeNoClone(toCopy->engine) : engine->Compose(toCopy->engine);
    } else {
        toRet = stabilizer->Compose(toCopy->stabilizer, qubitCount);
        ancillaCount += toCopy->ancillaCount;
    }

    // Resize the shards buffer.
    shards.insert(shards.begin() + qubitCount, toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (size_t i = qubitCount; i < shards.size(); ++i) {
        if (shards[i]) {
            shards[i] = shards[i]->Clone();
        }
    }

    SetQubitCount(nQubits);

    return toRet;
}

bitLenInt QStabilizerHybrid::Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    if (toCopy->ancillaCount) {
        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, qubitCount);
        const bitLenInt result = Compose(toCopy);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }

    const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
    bitLenInt toRet;

    if (engine) {
        toCopy->SwitchToEngine();
        toRet = engine->Compose(toCopy->engine, start);
    } else if (toCopy->engine) {
        SwitchToEngine();
        toRet = engine->Compose(toCopy->engine, start);
    } else {
        toRet = stabilizer->Compose(toCopy->stabilizer, start);
    }

    // Resize the shards buffer.
    shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (bitLenInt i = 0; i < toCopy->qubitCount; ++i) {
        if (shards[start + i]) {
            shards[start + i] = shards[start + i]->Clone();
        }
    }

    SetQubitCount(nQubits);

    return toRet;
}

QInterfacePtr QStabilizerHybrid::Decompose(bitLenInt start, bitLenInt length)
{
    QStabilizerHybridPtr dest = std::make_shared<QStabilizerHybrid>(engineTypes, length, 0, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);

    Decompose(start, dest);

    return dest;
}

void QStabilizerHybrid::Decompose(bitLenInt start, QStabilizerHybridPtr dest)
{
    const bitLenInt length = dest->qubitCount;

    if (!length) {
        return;
    }

    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        dest->SwitchToEngine();
        engine->Decompose(start, dest->engine);
        SetQubitCount(qubitCount - length);
        return;
    }

    if (dest->engine) {
        dest->engine.reset();
        dest->stabilizer = dest->MakeStabilizer(0U);
    }

    stabilizer->Decompose(start, dest->stabilizer);
    std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length)
{
    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        engine->Dispose(start, length);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    const bitLenInt nQubits = qubitCount - length;

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

bitLenInt QStabilizerHybrid::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QStabilizerHybridPtr nQubits = std::make_shared<QStabilizerHybrid>(cloneEngineTypes, length, 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);
    return Compose(nQubits, start);
}

void QStabilizerHybrid::GetQuantumState(complex* outputState)
{
    if (engine) {
        engine->GetQuantumState(outputState);
        return;
    }

    if (!ancillaCount && !IsBuffered()) {
        stabilizer->GetQuantumState(outputState);
        return;
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->SwitchToEngine();
    clone->GetQuantumState(outputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    if (engine) {
        engine->GetProbs(outputProbs);
        return;
    }

    if (!ancillaCount && !IsProbBuffered()) {
        stabilizer->GetProbs(outputProbs);
        return;
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->SwitchToEngine();
    clone->GetProbs(outputProbs);
}
complex QStabilizerHybrid::GetAmplitude(bitCapInt perm)
{
    if (engine) {
        return engine->GetAmplitude(perm);
    }

    if (!ancillaCount && !IsBuffered()) {
        return stabilizer->GetAmplitude(perm);
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->SwitchToEngine();
    return clone->GetAmplitude(perm);
}

void QStabilizerHybrid::SetQuantumState(complex const* inputState)
{
    DumpBuffers();

    if (qubitCount > 1U) {
        ancillaCount = 0;
        if (stabilizer) {
            engine = MakeEngine();
            stabilizer = NULL;
        }
        engine->SetQuantumState(inputState);

        return;
    }

    // Otherwise, we're preparing 1 qubit.
    engine = NULL;

    if (stabilizer && !ancillaCount) {
        stabilizer->SetPermutation(0U);
    } else {
        ancillaCount = 0;
        stabilizer = MakeStabilizer(0U);
    }

    const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
    const real1 sqrtProb = sqrt(prob);
    const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
    const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
    const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
    const complex mtrx[4U]{ sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
    Mtrx(mtrx, 0);
}

void QStabilizerHybrid::Mtrx(complex const* lMtrx, bitLenInt target)
{
    const bool wasCached = (bool)shards[target];
    complex mtrx[4U];
    if (wasCached) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4U, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4U, mtrx);
    }

    if (engine) {
        engine->Mtrx(mtrx, target);
        return;
    }

    if (IS_CLIFFORD(mtrx) || ((IS_PHASE(mtrx) || IS_INVERT(mtrx)) && stabilizer->IsSeparableZ(target))) {
        stabilizer->Mtrx(mtrx, target);
        return;
    }

    shards[target] = std::make_shared<MpsShard>(mtrx);
    if (!wasCached) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCMtrx(const std::vector<bitLenInt>& lControls, complex const* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(lControls, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(lControls, mtrx[1U], mtrx[2U], target);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls)) {
        return;
    }

    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->MCMtrx(lControls, mtrx, target);
}

void QStabilizerHybrid::MCPhase(
    const std::vector<bitLenInt>& lControls, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls)) {
        return;
    }

    if (!controls.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (stabilizer && (IS_NORM_0(topLeft - ONE_CMPLX) || IS_NORM_0(bottomRight - ONE_CMPLX))) {
        real1_f prob = Prob(target);
        if (IS_NORM_0(topLeft - ONE_CMPLX) && (prob == ZERO_R1)) {
            return;
        }
        if (IS_NORM_0(bottomRight - ONE_CMPLX) && (prob == ONE_R1)) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target, true);
    }

    if (engine) {
        engine->MCPhase(lControls, topLeft, bottomRight, target);
        return;
    }

    const bitLenInt control = controls[0U];
    stabilizer->MCPhase(controls, topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCInvert(
    const std::vector<bitLenInt>& lControls, complex topRight, complex bottomLeft, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls)) {
        return;
    }

    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (stabilizer && (controls.size() > 1U) && IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
        H(target);
        const real1_f prob = Prob(target);
        H(target);
        if (prob <= FP_NORM_EPSILON) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        engine->MCInvert(lControls, topRight, bottomLeft, target);
        return;
    }

    const bitLenInt control = controls[0U];
    stabilizer->MCInvert(controls, topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACMtrx(const std::vector<bitLenInt>& lControls, complex const* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(lControls, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(lControls, mtrx[1U], mtrx[2U], target);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->MACMtrx(lControls, mtrx, target);
}

void QStabilizerHybrid::MACPhase(
    const std::vector<bitLenInt>& lControls, complex topLeft, complex bottomRight, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (!controls.size()) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (stabilizer && (IS_NORM_0(topLeft - ONE_CMPLX) || IS_NORM_0(bottomRight - ONE_CMPLX))) {
        real1_f prob = Prob(target);
        if (IS_NORM_0(topLeft - ONE_CMPLX) && (prob == ZERO_R1)) {
            return;
        }
        if (IS_NORM_0(bottomRight - ONE_CMPLX) && (prob == ONE_R1)) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topLeft, bottomRight)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target, true);
    }

    if (engine) {
        engine->MACPhase(lControls, topLeft, bottomRight, target);
        return;
    }

    const bitLenInt control = controls[0U];
    stabilizer->MACPhase(controls, topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACInvert(
    const std::vector<bitLenInt>& lControls, complex topRight, complex bottomLeft, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, controls, true)) {
        return;
    }

    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if (stabilizer && (controls.size() > 1U) && IS_SAME(topRight, ONE_CMPLX) && IS_SAME(bottomLeft, ONE_CMPLX)) {
        H(target);
        const real1_f prob = Prob(target);
        H(target);
        if (prob <= FP_NORM_EPSILON) {
            return;
        }
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        engine->MACInvert(lControls, topRight, bottomLeft, target);
        return;
    }

    const bitLenInt control = controls[0U];
    stabilizer->MACInvert(controls, topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

real1_f QStabilizerHybrid::Prob(bitLenInt qubit)
{
    if (ancillaCount && !(stabilizer->IsSeparable(qubit))) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        clone->SwitchToEngine();
        return clone->Prob(qubit);
    }

    if (engine) {
        return engine->Prob(qubit);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shards[qubit] && !shards[qubit]->IsPhase()) {
        // Bit was already rotated to Z basis, if separable.
        if (stabilizer->IsSeparableZ(qubit)) {
            if (stabilizer->M(qubit)) {
                return (real1_f)norm(shards[qubit]->gate[3U]);
            }
            return (real1_f)norm(shards[qubit]->gate[2U]);
        }

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        return ONE_R1_F / 2;
    }

    if (stabilizer->IsSeparableZ(qubit)) {
        return stabilizer->M(qubit) ? ONE_R1_F : ZERO_R1_F;
    }

    // Otherwise, state appears locally maximally mixed.
    return ONE_R1_F / 2;
}

bool QStabilizerHybrid::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (ancillaCount && !(stabilizer->IsSeparable(qubit))) {
        SwitchToEngine();
    }

    if (engine) {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shards[qubit] && !shards[qubit]->IsPhase()) {
        if (stabilizer->IsSeparableZ(qubit)) {
            if (doForce) {
                if (doApply) {
                    if (result != stabilizer->ForceM(qubit, result, true, true)) {
                        // Sorry to throw, but the requested forced result is definitely invalid.
                        throw std::invalid_argument(
                            "QStabilizerHybrid::ForceM() forced a measurement result with 0 probability!");
                    }
                    shards[qubit] = NULL;
                }

                return result;
            }
            // Bit was already rotated to Z basis, if separable.
            return CollapseSeparableShard(qubit);
        }

        // Otherwise, we have non-Clifford measurement.
        SwitchToEngine();
        return engine->ForceM(qubit, result, doForce, doApply);
    }
    shards[qubit] = NULL;

    return stabilizer->ForceM(qubit, result, doForce, doApply);
}

bitCapInt QStabilizerHybrid::MAll()
{
    if (ancillaCount) {
        SwitchToEngine();
    }

    if (engine) {
        const bitCapInt toRet = engine->MAll();
        SetPermutation(toRet);
        return toRet;
    }

    bitCapInt toRet = 0U;
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i] && shards[i]->IsInvert()) {
            InvertBuffer(i);
        }

        if (shards[i] && !shards[i]->IsPhase()) {
            if (!stabilizer->IsSeparableZ(i)) {
                // Otherwise, we have non-Clifford measurement.
                SwitchToEngine();
                bitCapInt toRet = engine->MAll();
                SetPermutation(toRet);
                return toRet;
            }

            // Bit was already rotated to Z basis, if separable.
            CollapseSeparableShard(i);
        }
        shards[i] = NULL;

        if (stabilizer->M(i)) {
            toRet |= pow2(i);
        }
    }

    SetPermutation(toRet);

    return toRet;
}

std::map<bitCapInt, int> QStabilizerHybrid::MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    if (ancillaCount) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        clone->SwitchToEngine();
        return clone->MultiShotMeasureMask(qPowers, shots);
    }

    if (engine) {
        return engine->MultiShotMeasureMask(qPowers, shots);
    }

    std::vector<bitLenInt> bits(qPowers.size());
    std::transform(qPowers.begin(), qPowers.end(), bits.begin(), log2);

    std::map<bitCapInt, int> results;
    for (unsigned shot = 0U; shot < shots; ++shot) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (size_t i = 0U; i < qPowers.size(); ++i) {
            if (clone->M(bits[i])) {
                sample |= pow2(i);
            }
        }
        ++(results[sample]);
    }

    return results;
}

void QStabilizerHybrid::MultiShotMeasureMask(
    const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    if (ancillaCount) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        clone->SwitchToEngine();
        return clone->MultiShotMeasureMask(qPowers, shots, shotsArray);
    }

    if (engine) {
        engine->MultiShotMeasureMask(qPowers, shots, shotsArray);
        return;
    }

    std::vector<bitLenInt> bits(qPowers.size());
    std::transform(qPowers.begin(), qPowers.end(), bits.begin(), log2);

    par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (size_t i = 0U; i < qPowers.size(); ++i) {
            if (clone->M(bits[i])) {
                sample |= pow2(i);
            }
        }
        shotsArray[shot] = (unsigned)sample;
    });
}

real1_f QStabilizerHybrid::ApproxCompareHelper(QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol)
{
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
    QStabilizerHybridPtr thatClone =
        toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare->Clone()) : NULL;

    if (thisClone) {
        thisClone->FlushBuffers();
    }

    if (thatClone) {
        thatClone->FlushBuffers();
    }

    if (thisClone && thisClone->stabilizer && thatClone && thatClone->stabilizer) {
        if (isDiscreteBool) {
            return thisClone->stabilizer->ApproxCompare(thatClone->stabilizer, error_tol) ? ZERO_R1_F : ONE_R1_F;
        } else {
            return thisClone->stabilizer->SumSqrDiff(thatClone->stabilizer);
        }
    }

    if (thisClone) {
        thisClone->SwitchToEngine();
    }

    if (thatClone) {
        thatClone->SwitchToEngine();
    }

    QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
    QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

    const real1_f toRet = isDiscreteBool ? (thisEngine->ApproxCompare(thatEngine, error_tol) ? ZERO_R1_F : ONE_R1_F)
                                         : thisEngine->SumSqrDiff(thatEngine);

    if (toRet > TRYDECOMPOSE_EPSILON) {
        return toRet;
    }

    if (!stabilizer && toCompare->stabilizer) {
        SetPermutation(0U);
        stabilizer = std::dynamic_pointer_cast<QStabilizer>(toCompare->stabilizer->Clone());
        shards.resize(toCompare->shards.size());
        ancillaCount = toCompare->ancillaCount;
        for (size_t i = 0U; i < shards.size(); ++i) {
            shards[i] = toCompare->shards[i] ? toCompare->shards[i]->Clone() : NULL;
        }
    } else if (stabilizer && !toCompare->stabilizer) {
        toCompare->SetPermutation(0U);
        toCompare->stabilizer = std::dynamic_pointer_cast<QStabilizer>(stabilizer->Clone());
        toCompare->shards.resize(shards.size());
        toCompare->ancillaCount = ancillaCount;
        for (size_t i = 0U; i < shards.size(); ++i) {
            toCompare->shards[i] = shards[i] ? shards[i]->Clone() : NULL;
        }
    }

    return toRet;
}

void QStabilizerHybrid::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    if ((nrm > ZERO_R1) && (abs(ONE_R1 - nrm) > FP_NORM_EPSILON)) {
        SwitchToEngine();
    }

    if (stabilizer) {
        stabilizer->NormalizeState(REAL1_DEFAULT_ARG, norm_thresh, phaseArg);
    } else {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }
}

bool QStabilizerHybrid::TrySeparate(bitLenInt qubit)
{
    if ((qubitCount == 1U) && !ancillaCount) {
        return true;
    }

    if (stabilizer) {
        return stabilizer->CanDecomposeDispose(qubit, 1U);
    }

    return engine->TrySeparate(qubit);
}
bool QStabilizerHybrid::TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
{
    if ((qubitCount == 2U) && !ancillaCount) {
        return true;
    }

    if (engine) {
        return engine->TrySeparate(qubit1, qubit2);
    }

    if (qubit2 < qubit1) {
        std::swap(qubit1, qubit2);
    }

    stabilizer->Swap(qubit1 + 1U, qubit2);

    const bool toRet = stabilizer->CanDecomposeDispose(qubit1, 2U);

    stabilizer->Swap(qubit1 + 1U, qubit2);

    return toRet;
}
bool QStabilizerHybrid::TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol)
{
    if (engine) {
        return engine->TrySeparate(qubits, error_tol);
    }

    std::vector<bitLenInt> q(qubits.size());
    std::copy(qubits.begin(), qubits.end(), q.begin());
    std::sort(q.begin(), q.end());

    for (size_t i = 1U; i < q.size(); ++i) {
        Swap(q[0U] + i, q[i]);
    }

    const bool toRet = stabilizer->CanDecomposeDispose(q[0U], q.size());

    for (size_t i = 1U; i < q.size(); ++i) {
        Swap(q[0U] + i, q[i]);
    }

    return toRet;
}
} // namespace Qrack
