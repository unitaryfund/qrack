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
    , stateVecUnit(NULL)
    , maxQPowerOcl(pow2Ocl(qBitCount))
    , treeLevelCount(qBitCount)
    , attachedQubitCount(0)
    , bdtQubitCount(qBitCount)
    , shards(qBitCount)
{
#if ENABLE_PTHREAD
    SetConcurrency(std::thread::hardware_concurrency());
#endif
    SetPermutation(initState);
}

QInterfacePtr QBdt::MakeStateVector(bitLenInt qbCount, bitCapInt perm)
{
    return CreateQuantumInterface(engines, qbCount ? qbCount : qubitCount, perm, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, devID, hardware_rand_generator != NULL, false, amplitudeFloor);
}

bool QBdt::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    SetStateVector();
    bool toRet = stateVecUnit->ForceMParity(mask, result, doForce);
    ResetStateVector();

    return toRet;
}

void QBdt::SetPermutation(bitCapInt initState, complex phaseFac)
{
    DumpBuffers();
    Dump();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = ONE_CMPLX;
        }
    }

    root = std::make_shared<QBdtNode>(phaseFac);
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt qubit = 0; qubit < bdtQubitCount; qubit++) {
        const size_t bit = SelectBit(initState, qubit);
        leaf->branches[bit] = std::make_shared<QBdtNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }

    if (attachedQubitCount) {
        const size_t bit = SelectBit(initState, bdtQubitCount);
        leaf->branches[bit] = MakeQEngineNode(ONE_CMPLX, attachedQubitCount, initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = MakeQEngineNode(ZERO_CMPLX, 0);
    }
}

QInterfacePtr QBdt::Clone()
{
    FlushBuffers();

    QBdtPtr copyPtr = std::make_shared<QBdt>(bdtQubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->treeLevelCount = treeLevelCount;
    copyPtr->attachedQubitCount = attachedQubitCount;

    return copyPtr;
}

template <typename Fn> void QBdt::GetTraversal(Fn getLambda)
{
    Finish();

    for (bitCapIntOcl i = 0; i < GetMaxQPower(); i++) {
        QBdtNodeInterfacePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0; j < treeLevelCount; j++) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }

        if (!IS_NORM_0(scale) && attachedQubitCount) {
            scale *= std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->GetAmplitude(i >> bdtQubitCount);
        }

        getLambda((bitCapIntOcl)i, scale);
    }
}
template <typename Fn> void QBdt::SetTraversal(Fn setLambda)
{
    root = std::make_shared<QBdtNode>();

    for (bitCapIntOcl i = 0; i < GetMaxQPower(); i++) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < treeLevelCount; j++) {
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        setLambda((bitCapIntOcl)i, leaf);
    }

    root->ConvertStateVector(treeLevelCount);
    root->Prune(bdtQubitCount);
}
void QBdt::GetQuantumState(complex* state)
{
    FlushBuffers();
    GetTraversal([state](bitCapIntOcl i, complex scale) { state[i] = scale; });
}
void QBdt::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapIntOcl i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBdt::SetQuantumState(const complex* state)
{
    DumpBuffers();
    Dump();
    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, state](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->SetAmplitude(i >> qbCount, state[i]);
        } else {
            leaf->scale = state[i];
        }
    });
}
void QBdt::SetQuantumState(QInterfacePtr eng)
{
    Finish();
    const bool isAttached = attachedQubitCount;
    const bitLenInt qbCount = bdtQubitCount;
    SetTraversal([isAttached, qbCount, eng](bitCapIntOcl i, QBdtNodeInterfacePtr leaf) {
        if (isAttached) {
            std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->SetAmplitude(i >> qbCount, eng->GetAmplitude(i));
        } else {
            leaf->scale = eng->GetAmplitude(i);
        }
    });
}
void QBdt::GetProbs(real1* outputProbs)
{
    FlushBuffers();
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
    toCompare->FlushBuffers();

    complex projection = ZERO_CMPLX;
    for (bitCapIntOcl i = 0; i < maxQPowerOcl; i++) {
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
        scale *= std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->GetAmplitude(perm >> bdtQubitCount);
    }

    return scale;
}

bitLenInt QBdt::Compose(QBdtPtr toCopy, bitLenInt start)
{
    if (attachedQubitCount) {
        throw std::runtime_error("Compose() once attached is not implemented!");
    }

    if (start && (start != qubitCount)) {
        return QInterface::Compose(toCopy, start);
    }

    bitLenInt qbCount;
    bitCapIntOcl maxI;

    QBdtNodeInterfacePtr rootClone = toCopy->root->ShallowClone();
    if (start) {
        qbCount = bdtQubitCount;
        maxI = maxQPowerOcl;
    } else {
        qbCount = toCopy->bdtQubitCount;
        maxI = toCopy->maxQPowerOcl;
        root.swap(rootClone);
    }

    par_for_qbdt(0, maxI, [&](const bitCapIntOcl& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < qbCount; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(pow2Ocl(qbCount - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, qbCount - (j + 1U))];
        }

        if (!IS_NORM_0(leaf->scale)) {
            leaf->branches[0] = rootClone->branches[0];
            leaf->branches[1] = rootClone->branches[1];
        }

        return (bitCapIntOcl)0U;
    });

    shards.insert(shards.end(), toCopy->shards.begin(), toCopy->shards.end());

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

bitLenInt QBdt::Attach(QEnginePtr toCopy, bitLenInt start)
{
    if (start != qubitCount) {
        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, origSize);
        bitLenInt result = Attach(toCopy, qubitCount);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }

    const bool isAttached = attachedQubitCount;
    attachedQubitCount += toCopy->GetQubitCount();
    treeLevelCount = bdtQubitCount + 1U;

    if (isAttached) {
        par_for_qbdt(0, treeLevelPowerOcl, [&](const bitCapIntOcl& i, const int& cpu) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0; j < treeLevelCount; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapIntOcl)(pow2Ocl(treeLevelCount - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, treeLevelCount - (j + 1U))];
            }

            if (!IS_NORM_0(leaf->scale)) {
                std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->Compose(toCopy);
            }

            return (bitCapIntOcl)0U;
        });

        return start;
    }

    QEnginePtr toCopyQEngine = std::dynamic_pointer_cast<QEngine>(toCopy->Clone());

    const bitLenInt maxQubits = qubitCount - 1U;
    par_for_qbdt(0, pow2Ocl(maxQubits), [&](const bitCapIntOcl& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < maxQubits; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(pow2Ocl(maxQubits - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, maxQubits - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapIntOcl)0U;
        }

        for (size_t i = 0; i < 2; i++) {
            const complex scale = leaf->branches[i]->scale;
            if (IS_NORM_0(scale)) {
                leaf->branches[i] = MakeQEngineNode(ZERO_CMPLX, 0);
            } else {
                leaf->branches[i] = std::make_shared<QBdtQInterfaceNode>(scale, toCopyQEngine);
            }
        }

        return (bitCapIntOcl)0U;
    });

    return start;
}

void QBdt::DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest)
{
    if (attachedQubitCount) {
        throw std::runtime_error("Decompose() once attached is not implemented!");
    }

    const bitLenInt end = start + length;
    if ((attachedQubitCount && start) || (!attachedQubitCount && start && (end < qubitCount))) {
        ROR(start, 0, GetQubitCount());
        DecomposeDispose(0, length, dest);
        ROL(start, 0, GetQubitCount());

        return;
    }

    if (dest) {
        dest->DumpBuffers();
        dest->Dump();
    }

    const bool isReversed = !start;
    if (isReversed) {
        start = length;
        length = qubitCount - length;
    }

    bitCapIntOcl maxI = pow2Ocl(start);
    QBdtNodeInterfacePtr startNode = NULL;
    par_for_qbdt(0, maxI, [&](const bitCapIntOcl& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < start; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(pow2Ocl(start - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, start - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapIntOcl)0U;
        }

        if (!startNode) {
            // Whichever parallel write wins, this works.
            startNode = leaf->ShallowClone();
        }

        leaf->branches[0] = NULL;
        leaf->branches[1] = NULL;

        return (bitCapIntOcl)0U;
    });

    startNode->scale /= abs(startNode->scale);

    if (isReversed) {
        start = 0;
        length = qubitCount - length;
        root.swap(startNode);
    }

    if (dest) {
        dest->root = startNode;
        std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);

    SetQubitCount(qubitCount - length);
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? treeLevelCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

    FlushBuffer(qubit);

    std::map<QInterfacePtr, real1_f> qiProbs;

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
            // TODO: Is this right?
            QInterfacePtr qi = std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg;
            if (qiProbs.find(qi) == qiProbs.end()) {
                qiProbs[qi] = (real1_f)sqrt(
                    std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->Prob(qubit - bdtQubitCount));
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
    for (bitLenInt j = 0; j < treeLevelCount; j++) {
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    if (!IS_NORM_0(scale) && attachedQubitCount) {
        scale *= std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->GetAmplitude(perm >> bdtQubitCount);
    }

    return clampProb(norm(scale));
}

bool QBdt::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (doForce) {
        if (doApply) {
            ExecuteAsStateVector([&](QInterfacePtr eng) { eng->ForceM(qubit, result, true, doApply); });
        }
        return result;
    }

    const real1_f oneChance = Prob(qubit);
    if (oneChance >= ONE_R1) {
        result = true;
    } else if (oneChance <= ZERO_R1) {
        result = false;
    } else {
        result = (Rand() <= oneChance);
    }

    if (!doApply) {
        return result;
    }

    FlushBuffer(qubit);

    root->scale = GetNonunitaryPhase();

    const bitLenInt maxQubit = (qubit < bdtQubitCount) ? qubit : treeLevelCount;
    const bitCapInt qPower = pow2(maxQubit);

    for (bitCapInt i = 0; i < qPower; i++) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (IS_NORM_0(leaf->scale)) {
            continue;
        }

        if (maxQubit != qubit) {
            std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->ForceM(qubit - maxQubit, result, false, true);
            continue;
        }

        leaf->Branch();

        if (result) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale /= abs(leaf->branches[1]->scale);
        } else {
            leaf->branches[0]->scale /= abs(leaf->branches[0]->scale);
            leaf->branches[1]->SetZero();
        }
    }

    root->Prune(qubit + 1U);

    return result;
}

bitCapInt QBdt::MAll()
{
    FlushBuffers();

    bitCapInt result = 0;
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt i = 0; i < treeLevelCount; i++) {
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

        if (i >= bdtQubitCount) {
            result |= std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg->MAll() << bdtQubitCount;
            continue;
        }

        if (bitResult) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale = GetNonunitaryPhase();
            leaf = leaf->branches[1];
            result |= pow2(i);
        } else {
            leaf->branches[0]->scale = GetNonunitaryPhase();
            leaf->branches[1]->SetZero();
            leaf = leaf->branches[0];
        }
    }

    return result;
}

void QBdt::Apply2x2OnLeaf(const complex* mtrx, QBdtNodeInterfacePtr leaf, bitLenInt depth, bitCapInt highControlMask,
    bool isAnti, bool isParallel)
{
    // TODO: Finish pass through for ketControlsSorted to Attach() qubits.

    const bitLenInt remainder = bdtQubitCount - (depth + 1);
    leaf->Branch();

    QBdtNodeInterfacePtr& b0 = leaf->branches[0];
    QBdtNodeInterfacePtr& b1 = leaf->branches[1];

    const bitCapIntOcl maskTarget = (isAnti ? (bitCapIntOcl)0U : (bitCapIntOcl)highControlMask);

    IncrementFunc fn = [&](const bitCapIntOcl& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf0 = b0;
        QBdtNodeInterfacePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        // b0 and b1 can't both be 0.
        bool isZero = false;

        bitLenInt j;
        for (j = 0; j < remainder; j++) {
            leaf0->Branch(1, true);
            leaf1->Branch(1, true);

            const size_t bit = SelectBit(i, remainder - (j + 1U));

            leaf0 = leaf0->branches[bit];
            scale0 *= leaf0->scale;

            leaf1 = leaf1->branches[bit];
            scale1 *= leaf1->scale;

            isZero = IS_NORM_0(scale0) && IS_NORM_0(scale1);

            if (isZero) {
                break;
            }
        }

        if (isZero) {
            leaf0->SetZero();
            leaf1->SetZero();

            // WARNING: Mutates loop control variable!
            return (bitCapIntOcl)(pow2Ocl(remainder - (j + 1U)) - ONE_BCI);
        }

        if ((i & highControlMask) != maskTarget) {
            leaf0->scale = scale0;
            leaf1->scale = scale1;

            return (bitCapIntOcl)0U;
        }

        const complex Y0 = scale0;
        const complex Y1 = scale1;
        leaf0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        leaf1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        if (IS_NORM_0(leaf0->scale)) {
            leaf0->SetZero();
        }
        if (IS_NORM_0(leaf1->scale)) {
            leaf1->SetZero();
        }

        return (bitCapIntOcl)0U;
    };

    const bitCapIntOcl remainderPow = pow2Ocl(remainder);
    for (bitCapIntOcl i = 0; i < remainderPow; i++) {
        i |= fn(i, 0);
    }

    b0->ConvertStateVector(remainder);
    b1->ConvertStateVector(remainder);
    leaf->Prune(remainder + 1U);
}

template <typename Fn> void QBdt::ApplySingle(const complex* lMtrx, bitLenInt target, Fn leafFunc)
{
    const bitCapIntOcl targetPow = pow2Ocl(target);
    std::shared_ptr<complex> mtrx(new complex[4], std::default_delete<complex[]>());
    std::copy(lMtrx, lMtrx + 4U, mtrx.get());

    Dispatch(targetPow, [this, mtrx, target, targetPow, leafFunc]() {
        const bool isParallel = (pow2Ocl(target) < GetStride());

        par_for_qbdt(0, targetPow, [&](const bitCapIntOcl& i, const int& cpu) {
            QBdtNodeInterfacePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapIntOcl)(pow2Ocl(target - j) - ONE_BCI);
                }
                leaf->Branch();
                leaf = leaf->branches[SelectBit(i, target - (j + 1U))];
            }

            if (!IS_NORM_0(leaf->scale)) {
                leafFunc(leaf, mtrx.get(), 0U, isParallel);
            }

            return (bitCapIntOcl)0U;
        });

        root->Prune(target);
    });
}

void QBdt::Mtrx(const complex* lMtrx, bitLenInt target)
{
    // Attached qubits don't buffer
    if (target >= bdtQubitCount) {
        const bitCapInt qPower = pow2((target < bdtQubitCount) ? target : treeLevelCount);
        std::set<QInterfacePtr> qis;
        for (bitCapInt i = 0; i < qPower; i++) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0; j < treeLevelCount; j++) {
                if (!leaf) {
                    break;
                }
                leaf = leaf->branches[SelectBit(i, j)];
            }

            if (!leaf) {
                continue;
            }

            QInterfacePtr qiLeaf = std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg;
            if (qis.find(qiLeaf) == qis.end()) {
                qiLeaf->Mtrx(lMtrx, target - bdtQubitCount);
                qis.insert(qiLeaf);
            }
        }

        return;
    }

    complex mtrx[4];
    if (shards[target]) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        Phase(mtrx[0], mtrx[3], target);
        return;
    }
    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        Invert(mtrx[1], mtrx[2], target);
        return;
    }

    shards[target] = std::make_shared<MpsShard>(mtrx);
}

void QBdt::Phase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if ((target >= bdtQubitCount) || shards[target]) {
        Mtrx(mtrx, target);
        return;
    }

    if (IS_NORM_0(topLeft - bottomRight) && (randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft))) {
        return;
    }

    ApplySingle(mtrx, target, [](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl ignored, bool ignored2) {
        leaf->Branch();
        leaf->branches[0]->scale *= mtrx[0];
        leaf->branches[1]->scale *= mtrx[3];
        leaf->Prune();
    });
}

void QBdt::Invert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if ((target >= bdtQubitCount) || shards[target]) {
        Mtrx(mtrx, target);
        return;
    }

    ApplySingle(mtrx, target, [](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl ignored, bool ignored2) {
        leaf->Branch();
        leaf->branches[0].swap(leaf->branches[1]);
        leaf->branches[0]->scale *= mtrx[1];
        leaf->branches[1]->scale *= mtrx[2];
        leaf->Prune();
    });
}

template <typename Lfn>
void QBdt::ApplyControlledSingle(
    const complex* lMtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti, Lfn leafFunc)
{
    std::shared_ptr<complex> mtrxS(new complex[4], std::default_delete<complex[]>());
    std::copy(lMtrx, lMtrx + 4, mtrxS.get());

    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    std::vector<bitCapIntOcl> qPowersSorted;
    std::vector<bitLenInt> ketControlsVec;
    bitCapIntOcl lowControlMask = 0U;
    bitLenInt c;
    for (c = 0U; (c < controlLen) && (sortedControls[c] < target); c++) {
        if (sortedControls[c] > bdtQubitCount) {
            ketControlsVec.push_back(sortedControls[c]);
        }
        qPowersSorted.push_back(pow2Ocl(target - (sortedControls[c] + 1U)));
        lowControlMask |= qPowersSorted.back();
    }
    std::reverse(qPowersSorted.begin(), qPowersSorted.end());

    bitCapIntOcl highControlMask = 0U;
    for (; c < controlLen; c++) {
        if (sortedControls[c] > bdtQubitCount) {
            ketControlsVec.push_back(sortedControls[c]);
        }
        highControlMask |= pow2Ocl(bdtQubitCount - (sortedControls[c] + 1U));
    }

    const bool isKetSwapped =
        (lowControlMask || highControlMask) && (ketControlsVec.size() > 0) && (target < bdtQubitCount);
    if (isKetSwapped) {
        Swap(target, ketControlsVec[0]);
        std::swap(target, ketControlsVec[0]);
    }

    const bitCapIntOcl targetPow = pow2Ocl(target);
    const bitCapIntOcl maskTarget = (isAnti ? 0U : lowControlMask);

    Dispatch(targetPow,
        [this, mtrxS, target, targetPow, qPowersSorted, highControlMask, maskTarget, ketControlsVec, isKetSwapped,
            leafFunc]() {
            complex* mtrx = mtrxS.get();

            std::unique_ptr<bitLenInt[]> ketControls = NULL;
            if (ketControlsVec.size()) {
                ketControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[ketControlsVec.size()]);
                std::copy(ketControlsVec.begin(), ketControlsVec.end(), ketControls.get());
                for (bitLenInt i = 0U; i < ketControlsVec.size(); i++) {
                    ketControls[i] -= bdtQubitCount;
                }
            }

            if (qPowersSorted.size()) {
                root->Branch(target);
            }

            const bool isPhase = !highControlMask && IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]);
            const bool isInvert = !highControlMask && IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]);
            const bitCapIntOcl maxLcv = targetPow >> qPowersSorted.size();
            const bool isParallel = (maxLcv < GetStride());

            par_for_qbdt(0, maxLcv, [&](const bitCapIntOcl& lcv, const int& cpu) {
                bitCapIntOcl i = 0U;
                bitCapIntOcl iHigh = lcv;
                bitCapIntOcl iLow;
                int p;
                for (p = 0; p < (int)qPowersSorted.size(); p++) {
                    iLow = iHigh & (qPowersSorted[p] - ONE_BCI);
                    i |= iLow;
                    iHigh = (iHigh ^ iLow) << ONE_BCI;
                }
                i |= iHigh | maskTarget;

                QBdtNodeInterfacePtr leaf = root;
                // Iterate to qubit depth.
                for (bitLenInt j = 0; j < target; j++) {
                    if (IS_NORM_0(leaf->scale)) {
                        // WARNING: Mutates loop control variable!
                        i = pow2Ocl(target - j) - ONE_BCI;
                        for (p = (int)(qPowersSorted.size() - 1U); p >= 0; p--) {
                            i = (bitCapIntOcl)RemovePower(i, qPowersSorted[p]);
                        }
                        return i;
                    }
                    if (!qPowersSorted.size()) {
                        leaf->Branch();
                    }
                    leaf = leaf->branches[SelectBit(i, target - (j + 1U))];
                }

                if (IS_NORM_0(leaf->scale)) {
                    return (bitCapIntOcl)0U;
                }

                if (target >= bdtQubitCount) {
                    MCMtrx(ketControls.get(), ketControlsVec.size(), mtrxS.get(), target - bdtQubitCount);
                    return (bitCapIntOcl)0U;
                }

                if (isPhase) {
                    leaf->Branch();
                    leaf->branches[0]->scale *= mtrx[0];
                    leaf->branches[1]->scale *= mtrx[3];
                    leaf->Prune();
                } else if (isInvert) {
                    leaf->Branch();
                    leaf->branches[0].swap(leaf->branches[1]);
                    leaf->branches[0]->scale *= mtrx[1];
                    leaf->branches[1]->scale *= mtrx[2];
                    leaf->Prune();
                } else {
                    leafFunc(leaf, mtrx, highControlMask, isParallel);
                }

                return (bitCapIntOcl)0U;
            });

            root->Prune(target);

            // Undo isKetSwapped.
            if (isKetSwapped) {
                Swap(ketControlsVec[0], target);
            }
        });
}

void QBdt::MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (CheckControlled(controls, controlLen, mtrx, target, false)) {
        return;
    }

    ApplyControlledSingle(mtrx, controls, controlLen, target, false,
        [this, target](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl highControlMask, bool isParallel) {
            Apply2x2OnLeaf(mtrx, leaf, target, highControlMask, false, isParallel);
        });
}

void QBdt::MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (CheckControlled(controls, controlLen, mtrx, target, true)) {
        return;
    }

    ApplyControlledSingle(mtrx, controls, controlLen, target, true,
        [this, target](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl highControlMask, bool isParallel) {
            Apply2x2OnLeaf(mtrx, leaf, target, highControlMask, true, isParallel);
        });
}

bool QBdt::CheckControlled(
    const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target, bool isAnti)
{
    if (!controlLen) {
        Mtrx(mtrx, target);
        return true;
    }

    FlushBuffer(target);
    for (bitLenInt i = 0U; i < controlLen; i++) {
        FlushBuffer(controls[i]);
    }
    Finish();

    return false;
}

void QBdt::FlushBuffer(bitLenInt i)
{
    if (i >= bdtQubitCount) {
        return;
    }

    MpsShardPtr shard = shards[i];
    if (!shard) {
        return;
    }
    shards[i] = NULL;

    if (IS_NORM_0(shard->gate[1]) && IS_NORM_0(shard->gate[2])) {
        ApplySingle(
            shard->gate, i, [](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl ignored, bool ignored2) {
                leaf->Branch();
                leaf->branches[0]->scale *= mtrx[0];
                leaf->branches[1]->scale *= mtrx[3];
                leaf->Prune();
            });
        return;
    }

    if (IS_NORM_0(shard->gate[0]) && IS_NORM_0(shard->gate[3])) {
        ApplySingle(
            shard->gate, i, [](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl ignored, bool ignored2) {
                leaf->Branch();
                leaf->branches[0].swap(leaf->branches[1]);
                leaf->branches[0]->scale *= mtrx[1];
                leaf->branches[1]->scale *= mtrx[2];
                leaf->Prune();
            });
        return;
    }

    ApplySingle(shard->gate, i,
        [this, i](QBdtNodeInterfacePtr leaf, const complex* mtrx, bitCapIntOcl ignored, bool isParallel) {
            Apply2x2OnLeaf(mtrx, leaf, i, 0U, false, isParallel);
        });
}

} // namespace Qrack
