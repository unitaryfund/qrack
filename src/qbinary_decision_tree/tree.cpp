//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QEngineShard is the atomic qubit unit of the QUnit mapper. "PhaseShard" optimizations are basically just a very
// specific "gate fusion" type optimization, where multiple gates are composed into single product gates before
// application to the state vector, to reduce the total number of gates that need to be applied. Rather than handling
// this as a "QFusion" layer optimization, which will typically sit BETWEEN a base QEngine set of "shards" and a QUnit
// that owns them, this particular gate fusion optimization can be avoid representational entanglement in QUnit in the
// first place, which QFusion would not help with. Alternatively, another QFusion would have to be in place ABOVE the
// QUnit layer, (with QEngine "below,") for this to work. Additionally, QFusion is designed to handle more general gate
// fusion, not specifically controlled phase gates, which are entirely commuting among each other and possibly a
// jumping-off point for further general "Fourier basis" optimizations which should probably reside in QUnit, analogous
// to the |+>/|-> basis changes QUnit takes advantage of for "H" gates.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbinary_decision_tree.hpp"

namespace Qrack {

QBinaryDecisionTree::QBinaryDecisionTree(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , root(NULL)
{
    SetPermutation(initState);
}

bool QBinaryDecisionTree::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    QInterfacePtr copyPtr = std::make_shared<QEngineCPU>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

    GetQuantumState(copyPtr);
    bool toRet = copyPtr->ForceMParity(mask, result, doForce);
    SetQuantumState(copyPtr);

    return toRet;
}

real1_f QBinaryDecisionTree::ProbParity(const bitCapInt& mask)
{
    QInterfacePtr copyPtr = std::make_shared<QEngineCPU>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

    GetQuantumState(copyPtr);
    real1_f toRet = copyPtr->ProbParity(mask);
    SetQuantumState(copyPtr);

    return toRet;
}

void QBinaryDecisionTree::SetPermutation(bitCapInt initState, complex phaseFac)
{
    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = complex(ONE_R1, ZERO_R1);
        }
    }

    root = std::make_shared<QBinaryDecisionTreeNode>(phaseFac);
    QBinaryDecisionTreeNodePtr leaf = root;
    size_t bit;
    for (bitLenInt qubit = 0; qubit < qubitCount; qubit++) {
        bit = (initState >> qubit) & 1U;
        leaf->branches[bit] = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBinaryDecisionTreeNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }
}

QInterfacePtr QBinaryDecisionTree::Clone()
{
    QBinaryDecisionTreePtr copyPtr = std::make_shared<QBinaryDecisionTree>(qubitCount, 0, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf;
    QBinaryDecisionTreeNodePtr copyLeaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        copyLeaf = copyPtr->root;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
            copyLeaf = std::make_shared<QBinaryDecisionTreeNode>(leaf->scale);
        }
    }

    return copyPtr;
}

template <typename Fn> void QBinaryDecisionTree::GetTraversal(Fn getLambda)
{
    complex scale;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        scale = leaf->scale;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
            scale *= leaf->scale;
        }
        getLambda(i, scale);
    }
}
template <typename Fn> void QBinaryDecisionTree::SetTraversal(Fn setLambda)
{
    root = std::make_shared<QBinaryDecisionTreeNode>();
    root->Branch(qubitCount);

    bitCapInt maxQPower = pow2(qubitCount);
    bitLenInt j;

    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
        }
        setLambda(i, leaf);
    }

    root->Prune();
}
template <typename Fn> void QBinaryDecisionTree::ProductSetTraversal(Fn setLambda)
{
    if (IS_NORM_0(root->scale)) {
        // The tree isn't normalized, in this case, but this is defensive.
        return;
    }

    bitCapInt maxQPower = pow2(qubitCount);
    bitLenInt j;

    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        leaf->Branch();
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (IS_NORM_0(leaf->scale)) {
                break;
            }
            leaf->Branch();
        }
        if (!IS_NORM_0(leaf->scale)) {
            setLambda(i, leaf);
        }
    }

    root->Prune();
}
void QBinaryDecisionTree::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapInt i, complex scale) { state[i] = scale; });
}
void QBinaryDecisionTree::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapInt i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBinaryDecisionTree::SetQuantumState(const complex* state)
{
    SetTraversal([state](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = state[i]; });
}
void QBinaryDecisionTree::SetQuantumState(QInterfacePtr eng)
{
    SetTraversal([eng](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = eng->GetAmplitude(i); });
}
void QBinaryDecisionTree::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapInt i, complex scale) { outputProbs[i] = norm(scale); });
}

real1_f QBinaryDecisionTree::SumSqrDiff(QBinaryDecisionTreePtr toCompare)
{
    real1 projection = 0;

    complex scale1, scale2;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf1, leaf2;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf1 = root;
        leaf2 = toCompare->root;
        scale1 = leaf1->scale;
        scale2 = leaf2->scale;
        for (j = 0; j < qubitCount; j++) {
            leaf1 = leaf1->branches[(i >> j) & 1U];
            if (!leaf1) {
                break;
            }
            scale1 *= leaf1->scale;
        }
        for (j = 0; j < qubitCount; j++) {
            leaf2 = leaf2->branches[(i >> j) & 1U];
            if (!leaf2) {
                break;
            }
            scale2 *= leaf2->scale;
        }
        projection += norm(conj(scale2) * scale1);
    }

    return clampProb(ONE_R1 - projection);
}

complex QBinaryDecisionTree::GetAmplitude(bitCapInt perm)
{
    complex scale;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf = root;
    scale = leaf->scale;
    for (j = 0; j < qubitCount; j++) {
        leaf = leaf->branches[(perm >> j) & 1U];
        if (!leaf) {
            break;
        }
        scale *= leaf->scale;
    }

    return scale;
}
void QBinaryDecisionTree::SetAmplitude(bitCapInt perm, complex amp)
{
    int bit = 0;
    complex scale;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf = root;
    QBinaryDecisionTreeNodePtr child;
    scale = leaf->scale;
    for (j = 0; j < qubitCount; j++) {
        child = leaf;
        bit = (perm >> j) & 1U;
        child = leaf->branches[bit];
        if (!child) {
            child = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
            leaf->branches[bit] = child;
        } else {
            scale *= leaf->scale;
        }
    }

    child->scale = amp / scale;

    if (IS_NORM_0(child->scale - leaf->branches[bit ^ 1U]->scale)) {
        root->Prune(perm);
    }
}

bitLenInt QBinaryDecisionTree::Compose(QBinaryDecisionTreePtr toCopy, bitLenInt start)
{
    if (start == 0) {
        QBinaryDecisionTreePtr clone = std::dynamic_pointer_cast<QBinaryDecisionTree>(toCopy->Clone());
        std::swap(root, clone->root);
        clone->SetQubitCount(qubitCount);
        SetQubitCount(toCopy->qubitCount);
        ProductSetTraversal([clone](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) {
            QBinaryDecisionTreePtr toCopyClone = std::dynamic_pointer_cast<QBinaryDecisionTree>(clone->Clone());
            leaf->scale *= toCopyClone->root->scale;
        });
    } else if (start == qubitCount) {
        ProductSetTraversal([toCopy](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) {
            QBinaryDecisionTreePtr toCopyClone = std::dynamic_pointer_cast<QBinaryDecisionTree>(toCopy->Clone());
            leaf->scale *= toCopyClone->root->scale;
        });
    } else {
        throw std::runtime_error("Mid-range QBinaryDecisionTree::Compose() not yet implemented.");
    }

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}
void QBinaryDecisionTree::DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    bitLenInt i, j;
    QBinaryDecisionTreeNodePtr leaf;
    QBinaryDecisionTreeNodePtr child = root;
    for (i = 0; i < start; i++) {
        leaf = child;
        child = leaf->branches[0];
        if (!child) {
            // All amplitudes must be the same.
            if (dest) {
                dest->root->scale = randGlobalPhase ? std::polar(ONE_R1, 2 * PI_R1 * Rand()) : ONE_CMPLX;
            }
            break;
        }
    }

    // ANY child tree at this depth is assumed to be EXACTLY EQUIVALENT for the length to Decompose().
    // WARNING: "Compose()" doesn't seem to need a normalization pass, but does this?

    if (child && dest) {
        dest->root = child;
        dest->SetQubitCount(qubitCount - start);
        QBinaryDecisionTreePtr clone = std::dynamic_pointer_cast<QBinaryDecisionTree>(dest->Clone());
        dest->root = clone->root;

        if ((start + length) < qubitCount) {
            dest->Dispose(start + length, qubitCount - (start + length));
        }
    }

    complex scale;
    QBinaryDecisionTreeNodePtr remainderLeaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        scale = leaf->scale;
        for (j = 0; j < start; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
        }
        if (!leaf) {
            continue;
        }
        remainderLeaf = leaf;
        for (j = 0; j < length; j++) {
            remainderLeaf = remainderLeaf->branches[(i >> (start + j)) & 1U];
            if (!remainderLeaf) {
                break;
            }
        }
        if (!remainderLeaf) {
            continue;
        }
        leaf->branches[0] = remainderLeaf->branches[0];
        leaf->branches[1] = remainderLeaf->branches[1];
        leaf->scale = remainderLeaf->scale;
    }

    SetQubitCount(qubitCount - length);
}

real1_f QBinaryDecisionTree::Prob(bitLenInt qubitIndex)
{
    bitCapInt qPower = pow2(qubitIndex);
    bitCapInt qMask = qPower - ONE_BCI;
    bitCapInt maxLcv = maxQPower >> ONE_BCI;

    real1 prob = ZERO_R1;
    complex scale;
    bitCapInt i;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt lcv = 0; lcv < maxLcv; lcv++) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << ONE_BCI) | qPower;

        leaf = root;
        scale = leaf->scale;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
            scale *= leaf->scale;
        }
        prob += norm(scale);
    }

    return (real1)prob;
}

real1_f QBinaryDecisionTree::ProbAll(bitCapInt fullRegister)
{
    complex scale;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf = root;
    scale = leaf->scale;
    for (j = 0; j < qubitCount; j++) {
        leaf = leaf->branches[(fullRegister >> j) & 1U];
        if (!leaf) {
            break;
        }
        scale *= leaf->scale;
    }

    return clampProb(norm(scale));
}

bool QBinaryDecisionTree::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    real1_f oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            result = (Rand() <= oneChance);
        }
    }

    real1 nrmlzr;
    if (result) {
        nrmlzr = oneChance;
    } else {
        nrmlzr = ONE_R1 - oneChance;
    }

    if (nrmlzr <= ZERO_R1) {
        throw "ERROR: Forced a measurement result with 0 probability";
    }

    if (!doApply || (nrmlzr == ONE_R1)) {
        return result;
    }

    bitCapInt qPower = pow2(qubit);
    complex nrm = GetNonunitaryPhase() / (real1)(std::sqrt(nrmlzr));

    bitLenInt j;
    complex Y0;
    int bit;
    QBinaryDecisionTreeNodePtr leaf, child;
    for (bitCapInt i = 0; i < qPower; i++) {
        leaf = root;
        for (j = 0; j < qubit; j++) {
            bit = (i >> j) & 1U;
            child = leaf->branches[bit];
            if (!child) {
                child = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
                leaf->branches[bit] = child;
            }
            leaf = child;
        }
        leaf->Branch();

        if (result) {
            leaf->branches[0] = NULL;
            leaf->branches[1]->scale *= nrm;
        } else {
            leaf->branches[0]->scale *= nrm;
            leaf->branches[1] = NULL;
        }
    }

    root->Prune(qubit);

    return result;
}

void QBinaryDecisionTree::Apply2x2OnLeaves(
    const complex* mtrx, QBinaryDecisionTreeNodePtr* leaf0, QBinaryDecisionTreeNodePtr* leaf1)
{
    if (IS_NORM_0((*leaf0)->scale) && IS_NORM_0((*leaf1)->scale)) {
        (*leaf0)->scale = ZERO_CMPLX;
        (*leaf1)->scale = ZERO_CMPLX;
        return;
    }

    if (IS_NORM_0((*leaf0)->scale)) {
        (*leaf0) = (*leaf1)->DeepClone();
        (*leaf0)->scale = ZERO_CMPLX;
    }

    if (IS_NORM_0((*leaf1)->scale)) {
        (*leaf1) = (*leaf0)->DeepClone();
        (*leaf1)->scale = ZERO_CMPLX;
    }

    // Apply gate.
    complex Y0 = (*leaf0)->scale;
    (*leaf0)->scale = mtrx[0] * Y0 + mtrx[1] * (*leaf1)->scale;
    (*leaf1)->scale = mtrx[2] * Y0 + mtrx[3] * (*leaf1)->scale;
}

void QBinaryDecisionTree::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    root->Branch(target + 1U);

    bitLenInt j;
    int bit;
    bitCapInt qubitPower = pow2(target);
    QBinaryDecisionTreeNodePtr leaf, child;
    for (bitCapInt i = 0; i < qubitPower; i++) {
        // Iterate to qubit depth.
        leaf = root;
        for (j = 0; j < target; j++) {
            bit = (i >> j) & 1U;
            child = leaf->branches[bit];
            leaf = child;
        }

        Apply2x2OnLeaves(mtrx, &(leaf->branches[0]), &(leaf->branches[1]));
    }

    root->Prune(target);
}

void QBinaryDecisionTree::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (!controlLen) {
        ApplySingleBit(mtrx, target);
        return;
    }

    root->Branch(target + 1U);

    std::unique_ptr<bitLenInt[]> sortedControls(new bitLenInt[controlLen]);
    std::copy(controls, controls + controlLen, sortedControls.get());
    std::sort(sortedControls.get(), sortedControls.get() + controlLen);

    bitLenInt j;
    bitCapInt lowControlMask = 0;
    for (j = 0; (j < controlLen) && (sortedControls.get()[j] < target); j++) {
        lowControlMask |= pow2(sortedControls.get()[j]);
    }

    // "highControlMask" is only controls HIGHER than target, in the remaining body.
    bitCapInt highControlMask = 0;
    for (; j < controlLen; j++) {
        highControlMask |= pow2(sortedControls.get()[j]);
    }

    // The rest of the gate is only applying the INVERSE operation if control condition is NOT satisfied.

    // Consider CCNOT(0, 2, 1), (with target bit last). Draw a binary tree from root to 3 more levels down, (where
    // each branch from a node is a choice between |0> and |1> for the next-indexed qubit state). Order the
    // exponential rows by "control," "target", "control." Pointers have to be swapped and scaled across more than
    // immediate depth.

    complex invMtrx[4];
    inv2x2((complex*)mtrx, invMtrx);

    // A control occurs after the target, and we "push apart" by 1 bit.
    bitLenInt highBit =
        (target < sortedControls.get()[controlLen - 1U]) ? sortedControls.get()[controlLen - 1U] : target;
    bitCapInt targetPow = pow2(target);
    bitCapInt highControlPower = pow2(highBit - target) >> 1U;

    QBinaryDecisionTreeNodePtr parent, child0, child1;
    bitCapInt k, lcv2, bitPow;
    complex Y0;
    int bit;
    // Both the outer loop and the inner loop appear to be "embarrassingly parallel."
    for (bitCapInt i = 0; i < targetPow; i++) {
        // If any controls lower than the target aren't set, skip.
        if ((i & lowControlMask) != lowControlMask) {
            continue;
        }

        // Iterate to target bit.
        parent = root;
        for (j = 0; j < target; j++) {
            bit = (i >> j) & 1U;
            child0 = parent->branches[bit];
            parent = child0;
        }

        // All remaining controls have lower indices than the target.
        Apply2x2OnLeaves(mtrx, &(parent->branches[0]), &(parent->branches[1]));
        // (Consider "j" to be advanced by 1);

        if (!highControlMask) {
            continue;
        }

        parent->branches[0]->Branch();
        parent->branches[1]->Branch();

        // (The remainder is "embarrassingly parallel," from below this point.)
        for (lcv2 = 0; lcv2 < highControlPower; lcv2++) {
            k = i | (lcv2 << (target + 1U));

            // If all controls higher than the target are set, skip.
            if ((k & highControlMask) == highControlMask) {
                continue;
            }

            // Iterate for target bit.
            child0 = parent->branches[0];
            child1 = parent->branches[1];
            // (Children are already branched, to depth=1.)

            // Stay one bit advanced, for the last pair of children;
            bitPow = pow2(targetPow + 1U);
            bit = (k >> (target + 1U)) & 1U;

            // Starting where "j" left off, we trace the permutation for both children.
            // Break at first reset control bit, as we KNOW there is at least one reset control.
            for (j = (target + 2U); bit || !(bitPow & highControlMask); j++) {
                child0 = child0->branches[bit];
                child1 = child1->branches[bit];

                child0->Branch();
                child1->Branch();

                bitPow = pow2(j);
                bit = (k >> j) & 1U;
            }

            // Act inverse gate ONCE at LOWEST DEPTH that ANY control qubit is reset.
            if (k < bitPow) {
                Apply2x2OnLeaves(invMtrx, &(child0->branches[bit]), &(child1->branches[bit]));
            }
        }
    }

    root->Prune(highBit);
}

} // namespace Qrack
