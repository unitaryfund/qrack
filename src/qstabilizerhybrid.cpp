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
    (IS_SAME(mtrx[0], mtrx[1]) || IS_SAME(mtrx[0], -mtrx[1]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) ||                 \
        IS_SAME(mtrx[0], -I_CMPLX * mtrx[1])) &&                                                                       \
        (IS_SAME(mtrx[0], mtrx[2]) || IS_SAME(mtrx[0], -mtrx[2]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[2]) ||             \
            IS_SAME(mtrx[0], -I_CMPLX * mtrx[2])) &&                                                                   \
        (IS_SAME(mtrx[0], mtrx[3]) || IS_SAME(mtrx[0], -mtrx[3]) || IS_SAME(mtrx[0], I_CMPLX * mtrx[3]) ||             \
            IS_SAME(mtrx[0], -I_CMPLX * mtrx[3]))

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , engineTypes(eng)
    , engine(NULL)
    , shards(qubitCount)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , isSparse(useSparseStateVec)
    , isDefaultPaging(false)
    , separabilityThreshold(sep_thresh)
    , thresholdQubits(qubitThreshold)
    , maxPageQubits(-1)
    , deviceIDs(devList)
{
#if ENABLE_OPENCL
    if ((engineTypes.size() == 1U) && (engineTypes[0] == QINTERFACE_OPTIMAL_BASE)) {
        isDefaultPaging = true;
        bitLenInt segmentGlobalQb = 0U;
#if ENABLE_ENV_VARS
        if (getenv("QRACK_SEGMENT_GLOBAL_QB")) {
            segmentGlobalQb = (bitLenInt)std::stoi(std::string(getenv("QRACK_SEGMENT_GLOBAL_QB")));
        }
#endif

        DeviceContextPtr devContext = OCLEngine::Instance().GetDeviceContextPtr(devID);
        maxPageQubits = log2(devContext->GetMaxAlloc() / sizeof(complex)) - segmentGlobalQb;
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

void QStabilizerHybrid::CacheEigenstate(bitLenInt target)
{
    if (engine) {
        return;
    }

    MpsShardPtr toRet = NULL;
    // If in PauliX or PauliY basis, compose gate with conversion from/to PauliZ basis.
    if (stabilizer->IsSeparableX(target)) {
        // X eigenstate
        stabilizer->H(target);

        const complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(SQRT1_2_R1, ZERO_R1), complex(-SQRT1_2_R1, ZERO_R1) };
        toRet = std::make_shared<MpsShard>(mtrx);
    } else if (stabilizer->IsSeparableY(target)) {
        // Y eigenstate
        stabilizer->IS(target);
        stabilizer->H(target);

        const complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };
        toRet = std::make_shared<MpsShard>(mtrx);
    }

    if (!toRet) {
        return;
    }

    if (shards[target]) {
        toRet->Compose(shards[target]->gate);
    }

    shards[target] = toRet;

    if (IS_CLIFFORD(shards[target]->gate)) {
        MpsShardPtr shard = shards[target];
        shards[target] = NULL;
        Mtrx(shard->gate, target);
    }
}

QInterfacePtr QStabilizerHybrid::Clone()
{
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(engineTypes, qubitCount, 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int>{}, thresholdQubits, separabilityThreshold);

    Finish();
    c->Finish();

    if (stabilizer) {
        c->engine = NULL;
        c->stabilizer = std::dynamic_pointer_cast<QStabilizer>(stabilizer->Clone());
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (shards[i]) {
                c->shards[i] = std::make_shared<MpsShard>(shards[i]->gate);
            }
        }
    } else {
        // Clone and set engine directly.
        c->engine = std::dynamic_pointer_cast<QEngine>(engine->Clone());
        c->stabilizer = NULL;
    }

    return c;
}

QEnginePtr QStabilizerHybrid::CloneEmpty()
{
    QStabilizerHybridPtr c = std::make_shared<QStabilizerHybrid>(engineTypes, qubitCount, 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int>{}, thresholdQubits, separabilityThreshold);
    c->Finish();

    c->stabilizer = NULL;
    c->engine = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(engineTypes, 0, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam,
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

QInterfacePtr QStabilizerHybrid::Decompose(bitLenInt start, bitLenInt length)
{
    QStabilizerHybridPtr dest = std::make_shared<QStabilizerHybrid>(engineTypes, length, 0, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        std::vector<int>{}, thresholdQubits, separabilityThreshold);

    Decompose(start, dest);

    return dest;
}

void QStabilizerHybrid::Decompose(bitLenInt start, QStabilizerHybridPtr dest)
{
    const bitLenInt length = dest->qubitCount;
    const bitLenInt nQubits = qubitCount - length;
    const bool isPaging = isDefaultPaging && (nQubits <= maxPageQubits);

    if (length == qubitCount) {
        dest->stabilizer = stabilizer;
        stabilizer = NULL;
        dest->engine = engine;
        engine = NULL;

        dest->shards = shards;
        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        if (engineTypes[0] == QINTERFACE_QPAGER) {
            dest->TurnOnPaging();
        }
        dest->SwitchToEngine();
        engine->Decompose(start, dest->engine);
        if (isPaging) {
            TurnOffPaging();
        }
        SetQubitCount(qubitCount - length);
        return;
    }

    if (isPaging) {
        TurnOffPaging();
    }

    if (dest->engine) {
        dest->engine.reset();
        dest->stabilizer = dest->MakeStabilizer(0);
    }

    stabilizer->Decompose(start, dest->stabilizer);
    std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(nQubits);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length)
{
    const bitLenInt nQubits = qubitCount - length;
    const bool isPaging = isDefaultPaging && (nQubits <= maxPageQubits);

    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        engine->Dispose(start, length);
    } else {
        stabilizer->Dispose(start, length);
    }

    if (isPaging) {
        TurnOffPaging();
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    const bitLenInt nQubits = qubitCount - length;
    const bool isPaging = isDefaultPaging && (nQubits <= maxPageQubits);

    if (length == qubitCount) {
        stabilizer = NULL;
        engine = NULL;

        DumpBuffers();

        SetQubitCount(1);
        stabilizer = MakeStabilizer(0);
        return;
    }

    if (stabilizer && !stabilizer->CanDecomposeDispose(start, length)) {
        SwitchToEngine();
    }

    if (engine) {
        engine->Dispose(start, length, disposedPerm);
    } else {
        stabilizer->Dispose(start, length);
    }

    if (isPaging) {
        TurnOffPaging();
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);
}

void QStabilizerHybrid::SetQuantumState(const complex* inputState)
{
    DumpBuffers();

    if (qubitCount == 1U) {
        engine = NULL;

        if (stabilizer) {
            stabilizer->SetPermutation(0);
        } else {
            stabilizer = MakeStabilizer(0);
        }

        const real1 prob = (real1)clampProb(norm(inputState[1]));
        const real1 sqrtProb = sqrt(prob);
        const real1 sqrt1MinProb = (real1)sqrt(clampProb(ONE_R1 - prob));
        const complex phase0 = std::polar(ONE_R1, arg(inputState[0]));
        const complex phase1 = std::polar(ONE_R1, arg(inputState[1]));
        const complex mtrx[4] = { sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
        Mtrx(mtrx, 0);

        return;
    }

    SwitchToEngine();
    engine->SetQuantumState(inputState);
}

void QStabilizerHybrid::GetProbs(real1* outputProbs)
{
    FlushBuffers();

    if (stabilizer) {
        stabilizer->GetProbs(outputProbs);
    } else {
        engine->GetProbs(outputProbs);
    }
}

void QStabilizerHybrid::Mtrx(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4];
    if (shards[target]) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if (engine) {
        engine->Mtrx(mtrx, target);
        return;
    }

    try {
        stabilizer->Mtrx(mtrx, target);
    } catch (const std::domain_error&) {
        shards[target] = std::make_shared<MpsShard>(mtrx);
        CacheEigenstate(target);
    }
}

void QStabilizerHybrid::MCMtrx(const bitLenInt* lControls, bitLenInt lControlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MCPhase(lControls, lControlLen, mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MCInvert(lControls, lControlLen, mtrx[1], mtrx[2], target);
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

    if (controls.size() > 1U) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target, true);
    }

    if (engine) {
        engine->MCPhase(lControls, lControlLen, topLeft, bottomRight, target);
        return;
    }

    const bitLenInt control = controls[0];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    try {
        stabilizer->MCPhase(ctrls.get(), controls.size(), topLeft, bottomRight, target);
        if (shards[control]) {
            CacheEigenstate(control);
        }
        if (shards[target]) {
            CacheEigenstate(target);
        }
    } catch (const std::domain_error&) {
        SwitchToEngine();
        engine->MCPhase(lControls, lControlLen, topLeft, bottomRight, target);
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

    if (controls.size() > 1U) {
        SwitchToEngine();
    } else {
        FlushIfBlocked(controls[0], target);
    }

    if (engine) {
        engine->MCInvert(lControls, lControlLen, topRight, bottomLeft, target);
        return;
    }

    const bitLenInt control = controls[0];
    std::unique_ptr<bitLenInt[]> ctrls(new bitLenInt[controls.size()]);
    std::copy(controls.begin(), controls.end(), ctrls.get());
    try {
        stabilizer->MCInvert(ctrls.get(), controls.size(), topRight, bottomLeft, target);
        if (shards[control]) {
            CacheEigenstate(control);
        }
        if (shards[target]) {
            CacheEigenstate(target);
        }
    } catch (const std::domain_error&) {
        SwitchToEngine();
        engine->MCInvert(lControls, lControlLen, topRight, bottomLeft, target);
    }
}

void QStabilizerHybrid::MACMtrx(
    const bitLenInt* lControls, bitLenInt lControlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MACPhase(lControls, lControlLen, mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MACInvert(lControls, lControlLen, mtrx[1], mtrx[2], target);
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
        FlushIfBlocked(controls[0], target, true);
    }

    if (engine) {
        engine->MACPhase(lControls, lControlLen, topLeft, bottomRight, target);
        return;
    }

    X(controls[0]);
    MCPhase(&(controls[0]), 1U, topLeft, bottomRight, target);
    X(controls[0]);
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
        FlushIfBlocked(controls[0], target);
    }

    if (engine) {
        engine->MACInvert(lControls, lControlLen, topRight, bottomLeft, target);
        return;
    }

    X(controls[0]);
    MCInvert(&(controls[0]), 1U, topRight, bottomLeft, target);
    X(controls[0]);
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
                return norm(shards[qubit]->gate[3]);
            }
            return norm(shards[qubit]->gate[2]);
        }

        // Otherwise, buffer will not change the fact that state appears maximally mixed.
        return ONE_R1 / 2;
    }

    if (stabilizer->IsSeparableZ(qubit)) {
        return stabilizer->M(qubit) ? ONE_R1 : ZERO_R1;
    }

    // Otherwise, state appears locally maximally mixed.
    return ONE_R1 / 2;
}

bool QStabilizerHybrid::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (engine) {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    if (shards[qubit] && shards[qubit]->IsInvert()) {
        InvertBuffer(qubit);
    }

    // This check will first try to coax into decomposable form:
    if (doApply && !stabilizer->CanDecomposeDispose(qubit, 1)) {
        SwitchToEngine();
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    if (shards[qubit]) {
        if (!shards[qubit]->IsPhase() && stabilizer->IsSeparableZ(qubit)) {
            if (doForce) {
                if (doApply) {
                    if (result != stabilizer->M(qubit)) {
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
    bitCapInt toRet = 0;
    if (stabilizer) {
        for (bitLenInt i = 0; i < qubitCount; i++) {
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
    } else {
        toRet = engine->MAll();
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

    QStabilizerHybridPtr c = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    c->FlushBuffers();
    if (c->engine) {
        return c->engine->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }
    // Clear clone;
    c = NULL;

    std::vector<bitLenInt> bits(qPowerCount);
    for (bitLenInt i = 0U; i < qPowerCount; i++) {
        bits[i] = log2(qPowers[i]);
    }

    std::map<bitCapInt, int> results;
    for (unsigned shot = 0U; shot < shots; shot++) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (bitLenInt i = 0U; i < qPowerCount; i++) {
            if (clone->M(bits[i])) {
                sample |= pow2(i);
            }
        }
        results[sample]++;
    }

    return results;
}

void QStabilizerHybrid::MultiShotMeasureMask(
    const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned* shotsArray)
{
    if (!shots) {
        return;
    }

    if (engine) {
        engine->MultiShotMeasureMask(qPowers, qPowerCount, shots, shotsArray);
        return;
    }

    QStabilizerHybridPtr c = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
    c->FlushBuffers();
    if (c->engine) {
        c->engine->MultiShotMeasureMask(qPowers, qPowerCount, shots, shotsArray);
        return;
    }

    std::vector<bitLenInt> bits(qPowerCount);
    for (bitLenInt i = 0U; i < qPowerCount; i++) {
        bits[i] = log2(qPowers[i]);
    }

    par_for(0U, shots, [&](const bitCapIntOcl& shot, const unsigned& cpu) {
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        bitCapInt sample = 0U;
        for (bitLenInt i = 0U; i < qPowerCount; i++) {
            if (clone->M(bits[i])) {
                sample |= pow2(i);
            }
        }
        shotsArray[shot] = (unsigned)sample;
    });
}
} // namespace Qrack
