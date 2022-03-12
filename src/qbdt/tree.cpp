//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbdt_node.hpp"
#include "qfactory.hpp"

namespace Qrack {

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , engines(eng)
    , devID(deviceId)
    , root(NULL)
    , attachedQubitCount(0U)
    , bdtQubitCount(qBitCount)
    , bdtMaxQPower(pow2(qBitCount))
    , isAttached(false)
    , shards(qBitCount)
{
#if ENABLE_PTHREAD
    SetConcurrency(std::thread::hardware_concurrency());
#endif
    SetPermutation(initState);
}

QBdtQInterfaceNodePtr QBdt::MakeQInterfaceNode(complex scale, bitLenInt qbCount, bitCapInt perm)
{
    return std::make_shared<QBdtQInterfaceNode>(scale,
        CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, false,
            devID, hardware_rand_generator != NULL, false, amplitudeFloor));
}

void QBdt::FlushControlled(const bitLenInt* controls, bitLenInt controlLen, bitLenInt target)
{
    FlushBuffer(target);
    for (bitLenInt i = 0U; i < controlLen; i++) {
        FlushBuffer(controls[i]);
    }
}

void QBdt::FlushBuffer(bitLenInt i)
{
    MpsShardPtr shard = shards[i];
    if (!shard) {
        return;
    }
    shards[i] = NULL;

    ApplySingle(shard->gate, i);
}

void QBdt::FallbackMtrx(const complex* mtrx, bitLenInt target)
{
    if (!bdtQubitCount) {
        throw std::domain_error("QBdt has no universal qubits to fall back to, for FallbackMtrx()!");
    }

    bitLenInt randQb = bdtQubitCount * Rand();
    if (randQb >= bdtQubitCount) {
        randQb = bdtQubitCount;
    }

    Swap(randQb, target);
    Mtrx(mtrx, randQb);
    Swap(randQb, target);
}

void QBdt::FallbackMCMtrx(
    const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti)
{
    if (bdtQubitCount < (controlLen + 1U)) {
        throw std::domain_error("QBdt doesn't have enough universal qubits to fall back to, for FallbackMCMtrx()!");
    }

    bitLenInt randQb = (bdtQubitCount - controlLen) * Rand();
    if (randQb >= (bdtQubitCount - controlLen)) {
        randQb = (bdtQubitCount - controlLen);
    }

    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen]);
    for (bitLenInt i = 0U; i < controlLen; i++) {
        lControls[i] = randQb + i;
        Swap(randQb + i, controls[i]);
    }
    Swap(randQb + controlLen, target);

    ApplyControlledSingle(mtrx, lControls.get(), controlLen, controlLen, isAnti);

    Swap(randQb + controlLen, target);
    for (bitLenInt i = 0U; i < controlLen; i++) {
        Swap(controlLen - (randQb + i + 1U), controls[controlLen - (randQb + i + 1U)]);
    }
}

void QBdt::SetPermutation(bitCapInt initState, complex phaseFac)
{
    if (!qubitCount) {
        return;
    }

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = ONE_CMPLX;
        }
    }

    if (!bdtQubitCount) {
        root = MakeQInterfaceNode(phaseFac, attachedQubitCount, initState);

        return;
    }

    QInterfacePtr qReg = NULL;
    if (attachedQubitCount) {
        const bitCapInt maxI = pow2(bdtQubitCount);
        for (bitCapInt i = 0; i < maxI; i++) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0; j < bdtQubitCount; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    break;
                }
                leaf = leaf->branches[SelectBit(i, j)];
            }

            if (IS_NORM_0(leaf->scale)) {
                continue;
            }

            qReg = NODE_TO_QINTERFACE(leaf);
            if (!qReg) {
                break;
            }
        }
    }

    const bitLenInt maxQubit = attachedQubitCount ? (bdtQubitCount - 1U) : bdtQubitCount;
    root = std::make_shared<QBdtNode>(phaseFac);
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt qubit = 0; qubit < maxQubit; qubit++) {
        const size_t bit = SelectBit(initState, qubit);
        leaf->branches[bit] = std::make_shared<QBdtNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }

    if (attachedQubitCount) {
        const size_t bit = SelectBit(initState, maxQubit);
        leaf->branches[bit] = std::make_shared<QBdtQInterfaceNode>(ONE_CMPLX, qReg);
        qReg->SetPermutation(initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtQInterfaceNode>();
    }
}

QInterfacePtr QBdt::Clone()
{
    QBdtPtr copyPtr = std::make_shared<QBdt>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    ResetStateVector();

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->SetQubitCount(qubitCount, attachedQubitCount);
    copyPtr->isAttached = isAttached;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        copyPtr->shards[i] = shards[i] ? std::make_shared<MpsShard>(shards[i]->gate) : NULL;
    }

    return copyPtr;
}

template <typename Fn> void QBdt::GetTraversal(Fn getLambda)
{
    FlushBuffers();

    for (bitCapInt i = 0; i < bdtMaxQPower; i++) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0; j < bdtQubitCount; j++) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (!IS_NORM_0(scale) && attachedQubitCount) {
            scale *= NODE_TO_QINTERFACE(leaf)->GetAmplitude(i >> bdtQubitCount);
        }

        getLambda((bitCapIntOcl)i, scale);
    }
}
template <typename Fn> void QBdt::SetTraversal(Fn setLambda)
{
    root = std::make_shared<QBdtNode>();

    for (bitCapInt i = 0; i < bdtMaxQPower; i++) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < bdtQubitCount; j++) {
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }
        setLambda((bitCapIntOcl)i, leaf);
    }

    root->PopStateVector(bdtQubitCount);
    root->Prune(bdtQubitCount);
}
void QBdt::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapIntOcl i, complex scale) { state[i] = scale; });
}
void QBdt::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapIntOcl i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBdt::SetQuantumState(const complex* state)
{
    if (!bdtQubitCount) {
        NODE_TO_QINTERFACE(root)->SetQuantumState(state);
        return;
    }

    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, state](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            NODE_TO_QINTERFACE(leaf)->SetAmplitude(i >> qbCount, state[i]);
        } else {
            leaf->scale = state[i];
        }
    });
}
void QBdt::SetQuantumState(QInterfacePtr eng)
{
    if (!bdtQubitCount) {
        NODE_TO_QINTERFACE(root) = eng->Clone();
        return;
    }

    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, eng](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            NODE_TO_QINTERFACE(leaf)->SetAmplitude(i >> qbCount, eng->GetAmplitude(i));
        } else {
            leaf->scale = eng->GetAmplitude(i);
        }
    });
}
void QBdt::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapIntOcl i, complex scale) { outputProbs[i] = norm(scale); });
}

real1_f QBdt::SumSqrDiff(QBdtPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    FlushBuffers();
    ResetStateVector();
    toCompare->FlushBuffers();
    toCompare->ResetStateVector();

    complex projection = ZERO_CMPLX;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        QBdtNodeInterfacePtr leaf1 = root;
        QBdtNodeInterfacePtr leaf2 = toCompare->root;
        complex scale1 = leaf1->scale;
        complex scale2 = leaf2->scale;
        bitLenInt j;
        for (j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale1)) {
                break;
            }
            leaf1 = leaf1->branches[SelectBit(i, j)];
            scale1 *= leaf1->scale;
        }
        if (j < qubitCount) {
            continue;
        }
        for (j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale2)) {
                break;
            }
            leaf2 = leaf2->branches[SelectBit(i, j)];
            scale2 *= leaf2->scale;
        }
        if (j < qubitCount) {
            continue;
        }
        projection += conj(scale2) * scale1;
    }

    return ONE_R1 - clampProb(norm(projection));
}

complex QBdt::GetAmplitude(bitCapInt perm)
{
    FlushBuffers();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0; j < bdtQubitCount; j++) {
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NORM_0(scale) && attachedQubitCount) {
        scale *= NODE_TO_QINTERFACE(leaf)->GetAmplitude(perm >> bdtQubitCount);
    }

    return scale;
}

bitLenInt QBdt::Compose(QBdtPtr toCopy, bitLenInt start)
{
    ResetStateVector();
    toCopy->ResetStateVector();

    if (toCopy->attachedQubitCount) {
        throw std::domain_error("QBdt::Compose() not fully implemented, after Attach()!");
    }

    if (attachedQubitCount && start) {
        ROR(start, 0, qubitCount);
        Compose(toCopy, 0);
        ROL(start, 0, qubitCount);

        return start;
    }

    root->InsertAtDepth(toCopy->root, start, toCopy->bdtQubitCount);
    SetQubitCount(qubitCount + toCopy->qubitCount, attachedQubitCount + toCopy->attachedQubitCount);
    shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());

    return start;
}

bitLenInt QBdt::Attach(QStabilizerPtr toCopy)
{
    isAttached = true;
    const bitLenInt toRet = qubitCount;

    std::vector<MpsShardPtr> nShards(toCopy->GetQubitCount());
    shards.insert(shards.end(), nShards.begin(), nShards.end());

    if (!qubitCount) {
        QInterfacePtr toCopyClone = toCopy->Clone();
        root = std::make_shared<QBdtQInterfaceNode>(GetNonunitaryPhase(), toCopyClone);
        SetQubitCount(toCopy->GetQubitCount(), toCopy->GetQubitCount());

        return toRet;
    }

    if (attachedQubitCount) {
        par_for_qbdt(0, maxQPower, [&](const bitCapInt& i, const int& cpu) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0; j < bdtQubitCount; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapInt)(pow2(bdtQubitCount - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, bdtQubitCount - (j + 1U))];
            }

            if (!IS_NORM_0(leaf->scale)) {
                NODE_TO_QINTERFACE(leaf)->Compose(toCopy);
            }

            return (bitCapInt)0U;
        });

        SetQubitCount(qubitCount + toCopy->GetQubitCount(), attachedQubitCount + toCopy->GetQubitCount());

        return toRet;
    }

    QInterfacePtr toCopyClone = toCopy->Clone();

    const bitLenInt maxQubit = bdtQubitCount - 1U;
    const bitCapInt maxI = pow2(maxQubit);
    par_for_qbdt(0, maxI, [&](const bitCapInt& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        for (size_t j = 0; j < 2; j++) {
            const complex scale = leaf->branches[j]->scale;
            leaf->branches[j] = IS_NORM_0(scale) ? std::make_shared<QBdtQInterfaceNode>()
                                                 : std::make_shared<QBdtQInterfaceNode>(scale, toCopyClone);
        }

        return (bitCapInt)0U;
    });

    SetQubitCount(qubitCount + toCopy->GetQubitCount(), toCopy->GetQubitCount());

    return toRet;
}

QInterfacePtr QBdt::Decompose(bitLenInt start, bitLenInt length)
{
    QBdtPtr dest = std::make_shared<QBdt>(bdtQubitCount, length, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    Decompose(start, dest);

    return dest;
}

void QBdt::DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest)
{
    ResetStateVector();

    if (attachedQubitCount) {
        throw std::domain_error("QBdt::DecomposeDispose() not fully implemented, after Attach()!");
    }

    if (dest) {
        dest->ResetStateVector();
        dest->root = root->RemoveSeparableAtDepth(start, length);
    } else {
        root->RemoveSeparableAtDepth(start, length);
    }
    SetQubitCount(qubitCount - length, attachedQubitCount);
    shards.erase(shards.begin() + start, shards.begin() + start + length);

    root->Prune(bdtQubitCount);
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    FlushBuffer(qubit);

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

    std::map<QInterfacePtr, real1> qiProbs;

    real1 oneChance = ZERO_R1;
    for (bitCapInt i = 0; i < qPower; i++) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (IS_NORM_0(scale)) {
            continue;
        }

        if (isKet) {
            // Phase effects don't matter, for probability expectation.
            QInterfacePtr qi = NODE_TO_QINTERFACE(leaf);
            if (qiProbs.find(qi) == qiProbs.end()) {
                qiProbs[qi] = sqrt(NODE_TO_QINTERFACE(leaf)->Prob(qubit - bdtQubitCount));
            }
            oneChance += norm(scale * qiProbs[qi]);

            continue;
        }

        oneChance += norm(scale * leaf->branches[1]->scale);
    }

    return clampProb(oneChance);
}

real1_f QBdt::ProbAll(bitCapInt perm)
{
    FlushBuffers();

    QBdtNodeInterfacePtr leaf = root;
    complex scale = leaf->scale;
    for (bitLenInt j = 0; j < bdtQubitCount; j++) {
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NORM_0(scale) && attachedQubitCount) {
        scale *= NODE_TO_QINTERFACE(leaf)->GetAmplitude(perm >> bdtQubitCount);
    }

    return clampProb(norm(scale));
}

bool QBdt::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    const real1_f oneChance = Prob(qubit);
    if (oneChance >= ONE_R1) {
        result = true;
    } else if (oneChance <= ZERO_R1) {
        result = false;
    } else if (!doForce) {
        result = (Rand() <= oneChance);
    }

    if (!doApply) {
        return result;
    }

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);
    std::set<QInterfacePtr> qis;
    root->scale = GetNonunitaryPhase();

    for (bitCapInt i = 0; i < qPower; i++) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (IS_NORM_0(leaf->scale)) {
            continue;
        }

        if (isKet) {
            QInterfacePtr qi = NODE_TO_QINTERFACE(leaf);
            if (qis.find(qi) == qis.end()) {
                qis.insert(qi);
                qi->ForceM(qubit - bdtQubitCount, result, false, true);
            }
            continue;
        }

        leaf->Branch();

        QBdtNodeInterfacePtr& b0 = leaf->branches[0];
        QBdtNodeInterfacePtr& b1 = leaf->branches[1];

        if (result) {
            if (IS_NORM_0(b1->scale)) {
                throw std::runtime_error("ForceM() forced 0 probability!");
            }
            b0->SetZero();
            b1->scale /= abs(b1->scale);
        } else {
            if (IS_NORM_0(b0->scale)) {
                throw std::runtime_error("ForceM() forced 0 probability!");
            }
            b0->scale /= abs(b0->scale);
            b1->SetZero();
        }
    }

    root->Prune(maxQubit);

    return result;
}

bitCapInt QBdt::MAll()
{
    FlushBuffers();

    if (!bdtQubitCount) {
        const bitCapInt toRet = NODE_TO_QINTERFACE(root)->MAll();
        SetPermutation(toRet);

        return toRet;
    }

    bitCapInt result = 0;
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt i = 0; i < bdtQubitCount; i++) {
        leaf->Branch();
        real1_f oneChance = clampProb(norm(leaf->branches[1]->scale));
        bool bitResult;
        if (oneChance >= ONE_R1) {
            bitResult = true;
        } else if (oneChance <= ZERO_R1) {
            bitResult = false;
        } else {
            bitResult = (Rand() <= oneChance);
        }

        if (bitResult) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale = ONE_CMPLX;
            leaf = leaf->branches[1];
            result |= pow2(i);
        } else {
            leaf->branches[0]->scale = ONE_CMPLX;
            leaf->branches[1]->SetZero();
            leaf = leaf->branches[0];
        }
    }

    if (bdtQubitCount < qubitCount) {
        // Theoretically, there's only 1 copy of this leaf left, so no need to branch.
        result |= NODE_TO_QINTERFACE(leaf)->MAll() << bdtQubitCount;
    }

    return result;
}

void QBdt::ApplySingle(const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) && (randGlobalPhase || IS_NORM_0(ONE_CMPLX - mtrx[0])) &&
        IS_NORM_0(mtrx[0] - mtrx[3])) {
        return;
    }

    if (!bdtQubitCount) {
        NODE_TO_QINTERFACE(root)->Mtrx(mtrx, target);
        return;
    }

    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);

    std::set<QInterfacePtr> qis;
    bool isFail = false;

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0], mtrx[2]);
    const complex2 mtrxCol2(mtrx[1], mtrx[3]);
#endif

    par_for_qbdt(0, qPower, [&](const bitCapInt& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        // Iterate to qubit depth.
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        if (isKet) {
            QInterfacePtr qi = NODE_TO_QINTERFACE(leaf);
            if (qis.find(qi) == qis.end()) {
                try {
                    qi->Mtrx(mtrx, target - bdtQubitCount);
                } catch (const std::domain_error&) {
                    isFail = true;

                    return (bitCapInt)(qPower - ONE_BCI);
                }
                leaf->Prune();
                qis.insert(qi);
            }
        } else {
#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, bdtQubitCount - target);
#else
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
        }

        return (bitCapInt)0U;
    });

    if (!isFail) {
        root->Prune(maxQubit);

        return;
    }

    complex iMtrx[4];
    inv2x2(mtrx, iMtrx);
    std::set<QInterfacePtr>::iterator it = qis.begin();
    while (it != qis.end()) {
        (*it)->Mtrx(iMtrx, target - bdtQubitCount);
        it++;
    }
    root->Prune(maxQubit);

    FallbackMtrx(mtrx, target);
}

void QBdt::ApplyControlledSingle(
    const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti)
{
    FlushControlled(controls, controlLen, target);

    if (!bdtQubitCount) {
        if (isAnti) {
            NODE_TO_QINTERFACE(root)->MACMtrx(controls, controlLen, mtrx, target);
        } else {
            NODE_TO_QINTERFACE(root)->MCMtrx(controls, controlLen, mtrx, target);
        }
        return;
    }

    std::vector<bitLenInt> controlVec(controlLen);
    std::copy(controls, controls + controlLen, controlVec.begin());
    std::sort(controlVec.begin(), controlVec.end());
    const bool isSwapped = (target < controlVec.back()) && (target < bdtQubitCount);
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }

    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);
    std::vector<bitLenInt> ketControlsVec;
    bitCapInt lowControlMask = 0U;
    for (bitLenInt c = 0U; c < controlLen; c++) {
        const bitLenInt control = controlVec[c];
        if (control < bdtQubitCount) {
            lowControlMask |= pow2(maxQubit - (control + 1U));
        } else {
            ketControlsVec.push_back(control - bdtQubitCount);
        }
    }
    bitCapInt lowControlPerm = isAnti ? 0U : lowControlMask;
    std::unique_ptr<bitLenInt[]> ketControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[ketControlsVec.size()]);
    std::copy(ketControlsVec.begin(), ketControlsVec.end(), ketControls.get());
    std::set<QInterfacePtr> qis;

#if ENABLE_COMPLEX_X2
    const complex2 mtrxCol1(mtrx[0], mtrx[2]);
    const complex2 mtrxCol2(mtrx[1], mtrx[3]);
#endif
    bool isFail = false;

    par_for_qbdt(0, qPower, [&](const bitCapInt& i, const int& cpu) {
        if ((i & lowControlMask) != lowControlPerm) {
            return (bitCapInt)(lowControlMask - ONE_BCI);
        }

        QBdtNodeInterfacePtr leaf = root;
        // Iterate to qubit depth.
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(maxQubit - j) - ONE_BCI);
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        if (isKet) {
            QInterfacePtr qi = NODE_TO_QINTERFACE(leaf);
            if (qis.find(qi) == qis.end()) {
                try {
                    if (isAnti) {
                        qi->MACMtrx(ketControls.get(), ketControlsVec.size(), mtrx, target - bdtQubitCount);
                    } else {
                        qi->MCMtrx(ketControls.get(), ketControlsVec.size(), mtrx, target - bdtQubitCount);
                    }
                } catch (const std::domain_error&) {
                    isFail = true;

                    return (bitCapInt)(qPower - ONE_BCI);
                }
                leaf->Prune();
                qis.insert(qi);
            }
        } else {
#if ENABLE_COMPLEX_X2
            leaf->Apply2x2(mtrxCol1, mtrxCol2, bdtQubitCount - target);
#else
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
#endif
        }

        return (bitCapInt)0U;
    });

    if (!isFail) {
        root->Prune(maxQubit);
        // Undo isSwapped.
        if (isSwapped) {
            Swap(target, controlVec.back());
            std::swap(target, controlVec.back());
        }

        return;
    }

    complex iMtrx[4];
    inv2x2(mtrx, iMtrx);
    std::set<QInterfacePtr>::iterator it = qis.begin();
    while (it != qis.end()) {
        if (isAnti) {
            (*it)->MACMtrx(ketControls.get(), ketControlsVec.size(), iMtrx, target - bdtQubitCount);
        } else {
            (*it)->MCMtrx(ketControls.get(), ketControlsVec.size(), iMtrx, target - bdtQubitCount);
        }
        it++;
    }

    root->Prune(maxQubit);
    // Undo isSwapped.
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }

    FallbackMCMtrx(mtrx, controls, controlLen, target, isAnti);
}

void QBdt::Mtrx(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4];
    if (shards[target]) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if ((IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) || (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]))) {
        ApplySingle(mtrx, target);
        return;
    }

    shards[target] = std::make_shared<MpsShard>(mtrx);
}

void QBdt::MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (!controlLen) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MCPhase(controls, controlLen, mtrx[0], mtrx[3], target);
    } else if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MCInvert(controls, controlLen, mtrx[1], mtrx[2], target);
    } else {
        ApplyControlledSingle(mtrx, controls, controlLen, target, false);
    }
}

void QBdt::MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{

    if (!controlLen) {
        Mtrx(mtrx, target);
    } else if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MACPhase(controls, controlLen, mtrx[0], mtrx[3], target);
    } else if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MACInvert(controls, controlLen, mtrx[1], mtrx[2], target);
    } else {
        ApplyControlledSingle(mtrx, controls, controlLen, target, true);
    }
}

void QBdt::MCPhase(
    const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (!controlLen) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (!IS_NORM_0(ONE_CMPLX - topLeft)) {
        ApplyControlledSingle(mtrx, controls, controlLen, target, false);
        return;
    }

    std::unique_ptr<bitLenInt[]> lControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[controlLen]);
    std::copy(controls, controls + controlLen, lControls.get());
    std::sort(lControls.get(), lControls.get() + controlLen);

    if (target < lControls[controlLen - 1U]) {
        std::swap(target, lControls[controlLen - 1U]);
    }

    ApplyControlledSingle(mtrx, lControls.get(), controlLen, target, false);
}

void QBdt::MCInvert(
    const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    if (!controlLen) {
        Invert(topRight, bottomLeft, target);
        return;
    }

    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (!IS_NORM_0(ONE_CMPLX - topRight) || !IS_NORM_0(ONE_CMPLX - bottomLeft)) {
        ApplyControlledSingle(mtrx, controls, controlLen, target, false);
        return;
    }

    std::vector<bitLenInt> controlVec(controlLen);
    std::copy(controls, controls + controlLen, controlVec.begin());
    std::sort(controlVec.begin(), controlVec.end());

    if (controlVec.back() < target) {
        ApplyControlledSingle(mtrx, controls, controlLen, target, false);
        return;
    }

    H(target);
    MCPhase(controls, controlLen, ONE_CMPLX, -ONE_CMPLX, target);
    H(target);
}
} // namespace Qrack
