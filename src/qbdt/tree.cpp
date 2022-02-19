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

#define NODE_TO_QINTERFACE(leaf) std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg

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
    , attachedQubitCount(0)
    , bdtQubitCount(qBitCount)
{
#if ENABLE_PTHREAD
    SetConcurrency(std::thread::hardware_concurrency());
#endif
    SetPermutation(initState);
}

QInterfacePtr QBdt::MakeStateVector(bitLenInt qbCount, bitCapInt perm)
{
    return CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, devID, hardware_rand_generator != NULL, false, amplitudeFloor);
}

QBdtQInterfaceNodePtr QBdt::MakeQInterfaceNode(complex scale, bitLenInt qbCount, bitCapInt perm)
{
    return std::make_shared<QBdtQInterfaceNode>(scale,
        CreateQuantumInterface(engines, qbCount, perm, rand_generator, ONE_CMPLX, doNormalize, false, false, devID,
            hardware_rand_generator != NULL, false, amplitudeFloor));
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
    Dump();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = ONE_CMPLX;
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
        leaf->branches[bit] = MakeQInterfaceNode(ONE_CMPLX, attachedQubitCount, initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtQInterfaceNode>();
    }
}

QInterfacePtr QBdt::Clone()
{
    QBdtPtr copyPtr = std::make_shared<QBdt>(bdtQubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->attachedQubitCount = attachedQubitCount;
    copyPtr->SetQubitCount(qubitCount);

    return copyPtr;
}

template <typename Fn> void QBdt::GetTraversal(Fn getLambda)
{
    Finish();

    for (bitCapIntOcl i = 0; i < GetMaxQPower(); i++) {
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

    for (bitCapIntOcl i = 0; i < GetMaxQPower(); i++) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < bdtQubitCount; j++) {
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        setLambda((bitCapIntOcl)i, leaf);
    }

    root->ConvertStateVector(bdtQubitCount);
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
    Dump();
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
    Finish();
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

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

bitLenInt QBdt::Attach(QInterfacePtr toCopy)
{
    bitLenInt toRet = qubitCount;

    if (attachedQubitCount) {
        par_for_qbdt(0, maxQPowerOcl, [&](const bitCapIntOcl& i, const int& cpu) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0; j < bdtQubitCount; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return (bitCapIntOcl)(pow2Ocl(bdtQubitCount - j) - ONE_BCI);
                }
                leaf = leaf->branches[SelectBit(i, bdtQubitCount - (j + 1U))];
            }

            if (!IS_NORM_0(leaf->scale)) {
                NODE_TO_QINTERFACE(leaf)->Compose(toCopy);
            }

            return (bitCapIntOcl)0U;
        });

        attachedQubitCount += toCopy->GetQubitCount();
        SetQubitCount(bdtQubitCount + attachedQubitCount);

        return toRet;
    }

    QInterfacePtr toCopyClone = toCopy->Clone();

    const bitLenInt maxQubit = bdtQubitCount - 1U;
    const bitCapIntOcl maxI = pow2Ocl(maxQubit);
    par_for_qbdt(0, maxI, [&](const bitCapIntOcl& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(pow2Ocl(maxQubit - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapIntOcl)0U;
        }

        for (size_t i = 0; i < 2; i++) {
            const complex scale = leaf->branches[i]->scale;
            leaf->branches[i] = IS_NORM_0(scale) ? std::make_shared<QBdtQInterfaceNode>()
                                                 : std::make_shared<QBdtQInterfaceNode>(scale, toCopyClone);
        }

        return (bitCapIntOcl)0U;
    });

    attachedQubitCount = toCopy->GetQubitCount();
    SetQubitCount(bdtQubitCount + attachedQubitCount);

    return toRet;
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
    }

    SetQubitCount(qubitCount - length);
}

real1_f QBdt::Prob(bitLenInt qubit)
{
    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

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
            QInterfacePtr qi = NODE_TO_QINTERFACE(leaf);
            if (qiProbs.find(qi) == qiProbs.end()) {
                qiProbs[qi] = (real1_f)sqrt(NODE_TO_QINTERFACE(leaf)->Prob(qubit - bdtQubitCount));
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

    root->scale = GetNonunitaryPhase();

    const bitLenInt maxQubit = (qubit < bdtQubitCount) ? qubit : bdtQubitCount;
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
            NODE_TO_QINTERFACE(leaf)->ForceM(qubit - bdtQubitCount, result, false, true);
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
    bitCapInt result = 0;
    QBdtNodeInterfacePtr leaf = root;
    for (bitLenInt i = 0; i < qubitCount; i++) {
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
            result |= NODE_TO_QINTERFACE(leaf)->MAll() << bdtQubitCount;
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

void QBdt::Apply2x2OnLeaf(QBdtNodeInterfacePtr leaf, const complex* mtrx)
{
    leaf->Branch();
    QBdtNodeInterfacePtr& b0 = leaf->branches[0];
    QBdtNodeInterfacePtr& b1 = leaf->branches[1];

    const bool isZero0 = IS_NORM_0(b0->scale);
    const bool isZero1 = IS_NORM_0(b1->scale);

    const complex Y0 = b0->scale;
    b0->scale = mtrx[0] * Y0 + mtrx[1] * b1->scale;
    b1->scale = mtrx[2] * Y0 + mtrx[3] * b1->scale;

    if (isZero0 && !IS_NORM_0(b0->scale)) {
        b0->branches[0] = b1->branches[0];
        b0->branches[1] = b1->branches[1];
    }
    if (isZero1 && !IS_NORM_0(b1->scale)) {
        b1->branches[0] = b0->branches[0];
        b1->branches[1] = b0->branches[1];
    }

    if (IS_NORM_0(b0->scale)) {
        b0->SetZero();
    }
    if (IS_NORM_0(b1->scale)) {
        b1->SetZero();
    }

    leaf->Prune();
}

void QBdt::Mtrx(const complex* mtrx, bitLenInt target)
{
    if (target < bdtQubitCount) {
        const bitCapIntOcl targetPow = pow2Ocl(target);
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
                Apply2x2OnLeaf(leaf, mtrx);
            }

            return (bitCapIntOcl)0U;
        });
        root->Prune(target);

        return;
    }

    std::set<QInterfacePtr> qis;
    const bitCapInt qPower = pow2((target < bdtQubitCount) ? target : bdtQubitCount);
    par_for_qbdt(0, qPower, [&](const bitCapIntOcl& i, const int& cpu) {
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
        if (IS_NORM_0(leaf->scale)) {
            return (bitCapIntOcl)0U;
        }

        QInterfacePtr qiLeaf = NODE_TO_QINTERFACE(leaf);
        if (qis.find(qiLeaf) == qis.end()) {
            qiLeaf->Mtrx(mtrx, target - bdtQubitCount);
            qis.insert(qiLeaf);
        }

        return (bitCapIntOcl)0U;
    });
}

void QBdt::ApplyControlledSingle(const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target)
{
    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    const bool isSwapped = target < sortedControls.back();
    const bitLenInt swappedBit = target;
    if (isSwapped) {
        Swap(target, sortedControls.back());
        std::swap(target, sortedControls.back());
        std::sort(sortedControls.begin(), sortedControls.end());
    }

    const bitLenInt maxQubit = (target < bdtQubitCount) ? target : bdtQubitCount;
    const bitCapIntOcl maxQubitPow = pow2Ocl(maxQubit);
    std::vector<bitLenInt> ketControlsVec;
    bitCapIntOcl lowControlMask = 0U;
    for (bitLenInt c = 0U; c < controlLen; c++) {
        const bitLenInt control = sortedControls[c];
        if (control < bdtQubitCount) {
            lowControlMask |= pow2Ocl(maxQubit - (control + 1U));
        } else {
            ketControlsVec.push_back(control - bdtQubitCount);
        }
    }
    std::unique_ptr<bitLenInt[]> ketControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[ketControlsVec.size()]);
    std::copy(ketControlsVec.begin(), ketControlsVec.end(), ketControls.get());

    std::set<QInterfacePtr> qis;

    par_for_qbdt(0, maxQubitPow, [&](const bitCapIntOcl& i, const int& cpu) {
        if ((i & lowControlMask) != lowControlMask) {
            return (bitCapIntOcl)((lowControlMask ^ (i & lowControlMask)) - ONE_BCI);
        }

        QBdtNodeInterfacePtr leaf = root;
        // Iterate to qubit depth.
        for (bitLenInt j = 0; j < maxQubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(pow2Ocl(maxQubit - j) - ONE_BCI);
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, maxQubit - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapIntOcl)0U;
        }

        if (bdtQubitCount <= target) {
            QInterfacePtr qiLeaf = NODE_TO_QINTERFACE(leaf);
            if (qis.find(qiLeaf) == qis.end()) {
                qiLeaf->MCMtrx(ketControls.get(), ketControlsVec.size(), mtrx, target - bdtQubitCount);
                qis.insert(qiLeaf);
            }
        } else {
            Apply2x2OnLeaf(leaf, mtrx);
        }

        return (bitCapIntOcl)0U;
    });

    root->Prune(maxQubit);

    // Undo isSwapped.
    if (isSwapped) {
        Swap(swappedBit, target);
    }
}

void QBdt::MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (!controlLen) {
        Mtrx(mtrx, target);
        return;
    }

    if ((controlLen == 1U) && IS_NORM_0(ONE_CMPLX - mtrx[0]) && IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]) &&
        IS_NORM_0(ONE_CMPLX + mtrx[3])) {
        CZ(controls[0], target);
    } else if ((controlLen == 1U) && IS_NORM_0(mtrx[0]) && IS_NORM_0(ONE_CMPLX - mtrx[1]) &&
        IS_NORM_0(ONE_CMPLX - mtrx[2]) && IS_NORM_0(mtrx[3])) {
        CNOT(controls[0], target);
    } else {
        ApplyControlledSingle(mtrx, controls, controlLen, target);
    }
}

} // namespace Qrack
