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

#define NODE_TO_QINTERFACE(leaf) (std::dynamic_pointer_cast<QBdtQInterfaceNode>(leaf)->qReg)

namespace Qrack {

QBdt::QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , engines(eng)
    , devID(deviceId)
    , root(NULL)
    , stateVecUnit(NULL)
    , attachedQubitCount(0)
    , bdtQubitCount(qBitCount)
    , bdtMaxQPower(pow2(qBitCount))
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

QBdtQEngineNodePtr QBdt::MakeQEngineNode(complex scale, bitLenInt qbCount, bitCapInt perm)
{
    return std::make_shared<QBdtQEngineNode>(scale,
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
        leaf->branches[bit] = MakeQEngineNode(ONE_CMPLX, attachedQubitCount, initState >> bdtQubitCount);
        leaf->branches[bit ^ 1U] = std::make_shared<QBdtQEngineNode>();
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
    if (attachedQubitCount && (bdtQubitCount < start)) {
        const bitLenInt origBdtSize = bdtQubitCount;
        ROR(start - origBdtSize, 0, qubitCount);
        bitLenInt result = Compose(toCopy, origBdtSize);
        ROL(start - origBdtSize, 0, qubitCount);

        return result;
    }

    bitLenInt depth = start;
    bitLenInt size = toCopy->bdtQubitCount;
    root->InsertAtDepth(toCopy->root->ShallowClone(), depth, size);

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

bitLenInt QBdt::Attach(QEnginePtr toCopy)
{
    bitLenInt toRet = qubitCount;

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

        attachedQubitCount += toCopy->GetQubitCount();
        SetQubitCount(bdtQubitCount + attachedQubitCount);

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
            leaf->branches[j] = IS_NORM_0(scale) ? std::make_shared<QBdtQEngineNode>()
                                                 : std::make_shared<QBdtQEngineNode>(scale, toCopyClone);
        }

        return (bitCapInt)0U;
    });

    attachedQubitCount = toCopy->GetQubitCount();
    SetQubitCount(bdtQubitCount + attachedQubitCount);

    return toRet;
}

void QBdt::DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest)
{
    if (bdtQubitCount < length) {
        throw std::runtime_error("Decompose() once attached is not implemented for (bdtQubitCount < length)!");
    }

    if (start && (attachedQubitCount || ((start + length) != qubitCount))) {
        ROR(start, 0, qubitCount);
        DecomposeDispose(0, length, dest);
        ROL(start, 0, qubitCount);

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

    bitCapInt maxI = pow2(start);
    QBdtNodeInterfacePtr startNode = NULL;
    par_for_qbdt(0, maxI, [&](const bitCapInt& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf = root;
        for (bitLenInt j = 0; j < start; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(start - j) - ONE_BCI);
            }
            leaf = leaf->branches[SelectBit(i, start - (j + 1U))];
        }

        if (IS_NORM_0(leaf->scale)) {
            return (bitCapInt)0U;
        }

        if (!startNode) {
            // Whichever parallel write wins, this works.
            startNode = leaf->ShallowClone();
        }

        leaf->branches[0] = NULL;
        leaf->branches[1] = NULL;

        return (bitCapInt)0U;
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
            // TODO: Is this right?
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

    const bool isKet = (qubit >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : qubit;
    const bitCapInt qPower = pow2(maxQubit);

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

        leaf->Branch();

        if (isKet) {
            NODE_TO_QINTERFACE(leaf)->ForceM(qubit - bdtQubitCount, result, false, true);
            continue;
        }

        if (result) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale /= abs(leaf->branches[1]->scale);
        } else {
            leaf->branches[0]->scale /= abs(leaf->branches[0]->scale);
            leaf->branches[1]->SetZero();
        }
    }

    root->Prune(maxQubit + 1U);

    return result;
}

bitCapInt QBdt::MAll()
{
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

void QBdt::Mtrx(const complex* mtrx, bitLenInt target)
{
    const bool isKet = (target >= bdtQubitCount);
    const bitLenInt maxQubit = isKet ? bdtQubitCount : target;
    const bitCapInt qPower = pow2(maxQubit);

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
            leaf->Branch();
            NODE_TO_QINTERFACE(leaf)->Mtrx(mtrx, target - bdtQubitCount);
        } else {
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
        }

        return (bitCapInt)0U;
    });

    root->Prune(maxQubit + 1U);
}

void QBdt::ApplyControlledSingle(const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target)
{
    std::vector<bitLenInt> controlVec(controlLen);
    std::copy(controls, controls + controlLen, controlVec.begin());

    std::sort(controlVec.begin(), controlVec.end());
    const bool isSwapped = target < controlVec.back();
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
    std::unique_ptr<bitLenInt[]> ketControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[ketControlsVec.size()]);
    std::copy(ketControlsVec.begin(), ketControlsVec.end(), ketControls.get());

    par_for_qbdt(0, qPower, [&](const bitCapInt& i, const int& cpu) {
        if ((i & lowControlMask) != lowControlMask) {
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
            leaf->Branch();
            QInterfacePtr qiLeaf = NODE_TO_QINTERFACE(leaf);
            qiLeaf->MCMtrx(ketControls.get(), ketControlsVec.size(), mtrx, target - bdtQubitCount);
        } else {
            leaf->Apply2x2(mtrx, bdtQubitCount - target);
        }

        return (bitCapInt)0U;
    });

    root->Prune(maxQubit + 1U);

    // Undo isSwapped.
    if (isSwapped) {
        Swap(target, controlVec.back());
        std::swap(target, controlVec.back());
    }
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
        ApplyControlledSingle(mtrx, controls, controlLen, target);
    }
}

} // namespace Qrack
