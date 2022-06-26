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

#define IS_REAL_0(r) (abs(r) <= FP_NORM_EPSILON)
#define IS_CTRLED_CLIFFORD(top, bottom)                                                                                \
    ((IS_REAL_0(std::real(top)) || IS_REAL_0(std::imag(top))) && (IS_SAME(top, bottom) || IS_SAME(top, -bottom)))
#define IS_CLIFFORD(mtrx)                                                                                              \
    ((IS_SAME(mtrx[0U], mtrx[1U]) || IS_SAME(mtrx[0U], -mtrx[1U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) ||          \
         IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                                                                    \
        (IS_SAME(mtrx[0U], mtrx[2U]) || IS_SAME(mtrx[0U], -mtrx[2U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) ||       \
            IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                                                                 \
        (IS_SAME(mtrx[0U], mtrx[3U]) || IS_SAME(mtrx[0U], -mtrx[3U]) || IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) ||       \
            IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U])))
#define IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , isDefaultPaging(false)
    , doNormalize(doNorm)
    , isSparse(useSparseStateVec)
    , thresholdQubits(qubitThreshold)
    , maxPageQubits(-1)
    , separabilityThreshold(sep_thresh)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , engine(NULL)
    , deviceIDs(devList)
    , engineTypes(eng)
    , shards(qubitCount)
{
#if ENABLE_OPENCL
    if ((engineTypes.size() == 1U) && (engineTypes[0U] == QINTERFACE_OPTIMAL_BASE)) {
        isDefaultPaging = true;

        DeviceContextPtr devContext = OCLEngine::Instance().GetDeviceContextPtr(devID);
        maxPageQubits = log2(devContext->GetMaxAlloc() / sizeof(complex));
        if (qubitCount > maxPageQubits) {
            engineTypes.push_back(QINTERFACE_QPAGER);
        }
    }
#endif

    amplitudeFloor = REAL1_EPSILON;
    stabilizer = MakeStabilizer(initState);
}

QStabilizerPtr QStabilizerHybrid::MakeStabilizer(bitCapInt perm)
{
    return std::make_shared<QStabilizer>(
        qubitCount, perm, rand_generator, CMPLX_DEFAULT_ARG, false, randGlobalPhase, false, -1, useRDRAND);
}

QEnginePtr QStabilizerHybrid::MakeEngine(bitCapInt perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineTypes, qubitCount, perm, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
        thresholdQubits, separabilityThreshold);
    toRet->SetConcurrency(GetConcurrencyLevel());
    return std::dynamic_pointer_cast<QEngine>(toRet);
}

void QStabilizerHybrid::InvertBuffer(bitLenInt qubit)
{
    complex pauliX[4U] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
    pauliShard->Compose(shards[qubit]->gate);
    shards[qubit] = pauliShard->IsIdentity() ? NULL : pauliShard;
    stabilizer->X(qubit);
}

void QStabilizerHybrid::FlushH(bitLenInt qubit)
{
    complex hGate[4U] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
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
    if (shard && (!isPhase || !shard->IsPhase())) {
        SwitchToEngine();
    }
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
            SwitchToEngine();
        }
        return;
    }

    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        MpsShardPtr shard = shards[i];
        if (shard) {
            shards[i] = NULL;
            engine->Mtrx(shard->gate, i);
        }
    }
}

bool QStabilizerHybrid::TrimControls(
    const bitLenInt* lControls, bitLenInt lControlLen, std::vector<bitLenInt>& output, bool anti)
{
    if (engine) {
        output.insert(output.begin(), lControls, lControls + lControlLen);
        return false;
    }

    for (bitLenInt i = 0U; i < lControlLen; ++i) {
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
        const complex mtrx[4U] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(SQRT1_2_R1, ZERO_R1), complex(-SQRT1_2_R1, ZERO_R1) };
        toRet = std::make_shared<MpsShard>(mtrx);
    } else {
        stabilizer->H(target);
        stabilizer->IS(target);
        stabilizer->H(target);
        if (stabilizer->IsSeparableZ(target)) {
            // Y eigenstate
            const complex mtrx[4U] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
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
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(engineTypes, qubitCount, 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);

    Finish();
    c->Finish();

    if (engine) {
        // Clone and set engine directly.
        c->engine = std::dynamic_pointer_cast<QEngine>(engine->Clone());
        c->stabilizer = NULL;
        return c;
    }

    // Otherwise, stabilizer
    c->engine = NULL;
    c->stabilizer = std::dynamic_pointer_cast<QStabilizer>(stabilizer->Clone());
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (shards[i]) {
            c->shards[i] = std::make_shared<MpsShard>(shards[i]->gate);
        }
    }

    return c;
}

QEnginePtr QStabilizerHybrid::CloneEmpty()
{
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(engineTypes, qubitCount, 0U, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int64_t>{}, thresholdQubits, separabilityThreshold);
    c->Finish();

    c->stabilizer = NULL;
    c->engine = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(engineTypes, 0U, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam,
            devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits, separabilityThreshold));
    c->engine->SetConcurrency(GetConcurrencyLevel());

    c->engine->ZeroAmplitudes();
    c->engine->SetQubitCount(qubitCount);

    return c;
}

void QStabilizerHybrid::SwitchToEngine()
{
    if (engine) {
        return;
    }

    engine = MakeEngine();
    stabilizer->GetQuantumState(engine);
    stabilizer = NULL;
    FlushBuffers();
}

bitLenInt QStabilizerHybrid::Compose(QStabilizerHybridPtr toCopy)
{
    const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
    bitLenInt toRet;

    if (engine) {
        toCopy->SwitchToEngine();
        toRet = engine->Compose(toCopy->engine);
    } else if (toCopy->engine) {
        SwitchToEngine();
        toRet = engine->Compose(toCopy->engine);
    } else {
        toRet = stabilizer->Compose(toCopy->stabilizer);
    }

    // Resize the shards buffer.
    shards.insert(shards.end(), toCopy->shards.begin(), toCopy->shards.end());
    // Split the common shared_ptr references, with toCopy.
    for (bitLenInt i = qubitCount; i < nQubits; ++i) {
        if (shards[i]) {
            shards[i] = shards[i]->Clone();
        }
    }

    SetQubitCount(nQubits);

    return toRet;
}

bitLenInt QStabilizerHybrid::Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
{
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
    const bitLenInt nQubits = qubitCount - length;

    if (length == qubitCount) {
        dest->stabilizer = stabilizer;
        stabilizer = NULL;
        dest->engine = engine;
        engine = NULL;

        dest->shards = shards;
        DumpBuffers();

        SetQubitCount(1U);
        stabilizer = MakeStabilizer(0U);
        return;
    }

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

    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1U);
        stabilizer = MakeStabilizer(0U);
        return;
    }

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

    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1U);
        stabilizer = MakeStabilizer(0U);
        return;
    }

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::GetQuantumState(complex* outputState)
{
    if (engine) {
        engine->GetQuantumState(outputState);
        return;
    }

    if (!IsBuffered()) {
        stabilizer->GetQuantumState(outputState);
        return;
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->FlushBuffers();
    clone->GetQuantumState(outputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    if (engine) {
        engine->GetProbs(outputProbs);
        return;
    }

    if (!IsProbBuffered()) {
        stabilizer->GetProbs(outputProbs);
        return;
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->FlushBuffers();
    clone->GetProbs(outputProbs);
}
complex QStabilizerHybrid::GetAmplitude(bitCapInt perm)
{
    if (engine) {
        return engine->GetAmplitude(perm);
    }

    if (!IsBuffered()) {
        return stabilizer->GetAmplitude(perm);
    }

    QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    clone->FlushBuffers();
    return clone->GetAmplitude(perm);
}

void QStabilizerHybrid::SetQuantumState(const complex* inputState)
{
    DumpBuffers();

    if (qubitCount > 1U) {
        if (stabilizer) {
            engine = MakeEngine();
            stabilizer = NULL;
        }
        engine->SetQuantumState(inputState);

        return;
    }

    // Otherwise, we're preparing 1 qubit.
    engine = NULL;

    if (stabilizer) {
        stabilizer->SetPermutation(0U);
    } else {
        stabilizer = MakeStabilizer(0U);
    }

    const real1 prob = (real1)clampProb((real1_f)norm(inputState[1U]));
    const real1 sqrtProb = sqrt(prob);
    const real1 sqrt1MinProb = (real1)sqrt(clampProb((real1_f)(ONE_R1 - prob)));
    const complex phase0 = std::polar(ONE_R1, arg(inputState[0U]));
    const complex phase1 = std::polar(ONE_R1, arg(inputState[1U]));
    const complex mtrx[4U] = { sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
    Mtrx(mtrx, 0);
}

void QStabilizerHybrid::Mtrx(const complex* lMtrx, bitLenInt target)
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

void QStabilizerHybrid::MCMtrx(const bitLenInt* lControls, bitLenInt lControlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MCPhase(lControls, lControlLen, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MCInvert(lControls, lControlLen, mtrx[1U], mtrx[2U], target);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->MCMtrx(lControls, lControlLen, mtrx, target);
}

void QStabilizerHybrid::MCPhase(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - ONE_CMPLX) && IS_NORM_0(bottomRight - ONE_CMPLX)) {
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
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
        engine->MCPhase(lControls, lControlLen, topLeft, bottomRight, target);
        return;
    }

    const bitLenInt control = controls[0U];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    stabilizer->MCPhase(ctrls.get(), controls.size(), topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCInvert(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls)) {
        return;
    }

    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        engine->MCInvert(lControls, lControlLen, topRight, bottomLeft, target);
        return;
    }

    const bitLenInt control = controls[0U];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    stabilizer->MCInvert(ctrls.get(), controls.size(), topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACMtrx(
    const bitLenInt* lControls, bitLenInt lControlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        MACPhase(lControls, lControlLen, mtrx[0U], mtrx[3U], target);
        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        MACInvert(lControls, lControlLen, mtrx[1U], mtrx[2U], target);
        return;
    }

    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        Mtrx(mtrx, target);
        return;
    }

    SwitchToEngine();
    engine->MACMtrx(lControls, lControlLen, mtrx, target);
}

void QStabilizerHybrid::MACPhase(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
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
        engine->MACPhase(lControls, lControlLen, topLeft, bottomRight, target);
        return;
    }

    const bitLenInt control = controls[0U];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    stabilizer->MACPhase(ctrls.get(), controls.size(), topLeft, bottomRight, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MACInvert(
    const bitLenInt* lControls, bitLenInt lControlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    std::vector<bitLenInt> controls;
    if (TrimControls(lControls, lControlLen, controls, true)) {
        return;
    }

    if (!controls.size()) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    if ((controls.size() > 1U) || !IS_CTRLED_CLIFFORD(topRight, bottomLeft)) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0U], target);
    }

    if (engine) {
        engine->MACInvert(lControls, lControlLen, topRight, bottomLeft, target);
        return;
    }

    const bitLenInt control = controls[0U];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    stabilizer->MACInvert(ctrls.get(), controls.size(), topRight, bottomLeft, target);
    if (shards[control]) {
        CacheEigenstate(control);
    }
    if (shards[target]) {
        CacheEigenstate(target);
    }
}

real1_f QStabilizerHybrid::Prob(bitLenInt qubit)
{
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
    if (engine) {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    if (shards[qubit]) {
        if (!shards[qubit]->IsPhase() && stabilizer->IsSeparableZ(qubit)) {
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

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        shards[qubit] = NULL;
    }

    return stabilizer->ForceM(qubit, result, doForce, doApply);
}

bitCapInt QStabilizerHybrid::MAll()
{
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

        if (shards[i]) {
            if (!shards[i]->IsPhase() && stabilizer->IsSeparableZ(i)) {
                // Bit was already rotated to Z basis, if separable.
                CollapseSeparableShard(i);
            }

            // Otherwise, buffer will not change the fact that state appears maximally mixed.
            shards[i] = NULL;
        }

        if (stabilizer->M(i)) {
            toRet |= pow2(i);
        }
    }

    SetPermutation(toRet);

    return toRet;
}

std::map<bitCapInt, int> QStabilizerHybrid::MultiShotMeasureMask(
    const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    if (engine) {
        return engine->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }

    std::vector<bitLenInt> bits(qPowerCount);
    for (bitLenInt i = 0U; i < qPowerCount; ++i) {
        bits[i] = log2(qPowers[i]);
    }

    std::map<bitCapInt, int> results;
    for (unsigned shot = 0U; shot < shots; ++shot) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (bitLenInt i = 0U; i < qPowerCount; ++i) {
            if (clone->M(bits[i])) {
                sample |= pow2(i);
            }
        }
        ++(results[sample]);
    }

    return results;
}

void QStabilizerHybrid::MultiShotMeasureMask(
    const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned long long* shotsArray)
{
    if (!shots) {
        return;
    }

    if (engine) {
        engine->MultiShotMeasureMask(qPowers, qPowerCount, shots, shotsArray);
        return;
    }

    std::vector<bitLenInt> bits(qPowerCount);
    for (bitLenInt i = 0U; i < qPowerCount; ++i) {
        bits[i] = log2(qPowers[i]);
    }

    par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (bitLenInt i = 0U; i < qPowerCount; ++i) {
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
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            shards[i] = toCompare->shards[i] ? toCompare->shards[i]->Clone() : NULL;
        }
    } else if (stabilizer && !toCompare->stabilizer) {
        toCompare->SetPermutation(0U);
        toCompare->stabilizer = std::dynamic_pointer_cast<QStabilizer>(stabilizer->Clone());
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            toCompare->shards[i] = shards[i] ? shards[i]->Clone() : NULL;
        }
    }

    return toRet;
}

void QStabilizerHybrid::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    if (abs(nrm) <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
        return;
    }

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
    if (qubitCount == 1U) {
        return true;
    }

    if (stabilizer) {
        return stabilizer->CanDecomposeDispose(qubit, 1U);
    }

    return engine->TrySeparate(qubit);
}
bool QStabilizerHybrid::TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubitCount == 2U) {
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
bool QStabilizerHybrid::TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol)
{
    if (engine) {
        return engine->TrySeparate(qubits, length, error_tol);
    }

    std::vector<bitLenInt> q(length);
    std::copy(qubits, qubits + length, q.begin());
    std::sort(q.begin(), q.end());

    for (bitLenInt i = 1U; i < length; ++i) {
        Swap(q[0U] + i, q[i]);
    }

    const bool toRet = stabilizer->CanDecomposeDispose(q[0U], length);

    for (bitLenInt i = 1U; i < length; ++i) {
        Swap(q[0U] + i, q[i]);
    }

    return toRet;
}
} // namespace Qrack
