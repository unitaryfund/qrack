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

#include "qbinary_decision_tree.hpp"

#include <mutex>

namespace Qrack {

QBinaryDecisionTree::QBinaryDecisionTree(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , root(NULL)
{
    pStridePow =
        getenv("QRACK_PSTRIDEPOW") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW;
    SetConcurrency(std::thread::hardware_concurrency());
    SetPermutation(initState);
}

bool QBinaryDecisionTree::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    QEnginePtr copyPtr = std::make_shared<QEngineCPU>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

    GetQuantumState(copyPtr);
    bool toRet = copyPtr->ForceMParity(mask, result, doForce);
    SetQuantumState(copyPtr);

    return toRet;
}

void QBinaryDecisionTree::SetPermutation(bitCapInt initState, complex phaseFac)
{
    Dump();

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
    Finish();

    QBinaryDecisionTreePtr copyPtr = std::make_shared<QBinaryDecisionTree>(qubitCount, 0, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;

    return copyPtr;
}

template <typename Fn> void QBinaryDecisionTree::GetTraversal(Fn getLambda)
{
    Finish();

    par_for(0, maxQPower, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
            scale *= leaf->scale;
            if (IS_NORM_0(scale)) {
                break;
            }
        }
        getLambda(i, scale);
    });
}
template <typename Fn> void QBinaryDecisionTree::SetTraversal(Fn setLambda)
{
    Dump();

    root = std::make_shared<QBinaryDecisionTreeNode>();
    root->Branch(qubitCount);

    bitCapInt maxQPower = pow2(qubitCount);
    par_for(0, maxQPower, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
        }
        setLambda(i, leaf);
    });

    root->ConvertStateVector(qubitCount);
    root->scale = GetNonunitaryPhase();
    root->Prune(qubitCount);
}
void QBinaryDecisionTree::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapInt i, complex scale) { state[i] = scale; });
}
void QBinaryDecisionTree::GetQuantumState(QEnginePtr eng)
{
    GetTraversal([eng](bitCapInt i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBinaryDecisionTree::SetQuantumState(const complex* state)
{
    SetTraversal([state](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = state[i]; });
}
void QBinaryDecisionTree::SetQuantumState(QEnginePtr eng)
{
    SetTraversal([eng](bitCapInt i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = eng->GetAmplitude(i); });
}
void QBinaryDecisionTree::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapInt i, complex scale) { outputProbs[i] = norm(scale); });
}

real1_f QBinaryDecisionTree::SumSqrDiff(QBinaryDecisionTreePtr toCompare)
{
    Finish();

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> partInner(new complex[numCores]());

    par_for(0, maxQPower, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf1 = root;
        QBinaryDecisionTreeNodePtr leaf2 = toCompare->root;
        complex scale1 = leaf1->scale;
        complex scale2 = leaf2->scale;
        bitLenInt j;
        for (j = 0; j < qubitCount; j++) {
            leaf1 = leaf1->branches[(i >> j) & 1U];
            if (!leaf1) {
                break;
            }
            scale1 *= leaf1->scale;
            if (IS_NORM_0(scale1)) {
                return;
            }
        }
        for (j = 0; j < qubitCount; j++) {
            leaf2 = leaf2->branches[(i >> j) & 1U];
            if (!leaf2) {
                break;
            }
            scale2 *= leaf2->scale;
            if (IS_NORM_0(scale2)) {
                return;
            }
        }
        partInner[cpu] += conj(scale2) * scale1;
    });

    complex projection = 0;
    for (int i = 0; i < numCores; i++) {
        projection += partInner[i];
    }

    return ONE_R1 - clampProb(norm(projection));
}

complex QBinaryDecisionTree::GetAmplitude(bitCapInt perm)
{
    Finish();

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
        if (IS_NORM_0(scale)) {
            break;
        }
    }

    return scale;
}

bitLenInt QBinaryDecisionTree::Compose(QBinaryDecisionTreePtr toCopy, bitLenInt start)
{
    Finish();
    toCopy->Finish();

    if (start == 0) {
        QBinaryDecisionTreeNodePtr rootClone = toCopy->root->ShallowClone();
        std::swap(root, rootClone);
        par_for(0, toCopy->maxQPower, [&](const bitCapInt& i, const int& cpu) {
            QBinaryDecisionTreeNodePtr leaf = root;
            for (bitLenInt j = 0; j < toCopy->qubitCount; j++) {
                leaf = leaf->branches[(i >> j) & 1U];
                if (!leaf) {
                    return;
                }
            }
            leaf->branches[0] = rootClone->branches[0];
            leaf->branches[1] = rootClone->branches[1];
        });
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return start;
    }

    if (start == qubitCount) {
        par_for(0, maxQPower, [&](const bitCapInt& i, const int& cpu) {
            QBinaryDecisionTreeNodePtr leaf = root;
            for (bitLenInt j = 0; j < qubitCount; j++) {
                leaf = leaf->branches[(i >> j) & 1U];
                if (!leaf) {
                    return;
                }
            }
            leaf->branches[0] = toCopy->root->branches[0];
            leaf->branches[1] = toCopy->root->branches[1];
        });
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return start;
    }

    ExecuteAsQEngineCPU([&](QInterfacePtr eng) {
        QEnginePtr copyPtr = std::make_shared<QEngineCPU>(toCopy->qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

        toCopy->GetQuantumState(copyPtr);
        eng->Compose(copyPtr, start);
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SetQuantumState(copyPtr);
    });

    return start;
}
void QBinaryDecisionTree::DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    Finish();
    if (dest) {
        dest->Dump();
    }

    bitLenInt end = start + length;
    bitCapInt maxI = pow2(start);
    QBinaryDecisionTreeNodePtr startNode = NULL;
    std::mutex destMutex;
    par_for(0, maxI, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < start; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf || IS_NORM_0(leaf->scale)) {
                return;
            }
        }

        if (!leaf || IS_NORM_0(leaf->scale)) {
            return;
        }

        if (!startNode) {
            const std::lock_guard<std::mutex> destLock(destMutex);
            // Now that we've locked, is the startNode still not set?
            if (!startNode) {
                startNode = leaf->ShallowClone();
            }
        }

        leaf->branches[0] = NULL;
        leaf->branches[1] = NULL;
    });

    if (dest) {
        dest->root = startNode;
        dest->root->scale = GetNonunitaryPhase();
    }

    if (qubitCount <= end) {
        root->Prune(end);
        if (dest) {
            dest->root->Prune(length);
        }
        SetQubitCount(end);
        return;
    }

    bitCapInt lengthPow = pow2(length);
    QBinaryDecisionTreeNodePtr endNode = NULL;

    par_for(0, lengthPow, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = startNode;
        for (bitLenInt j = 0; j < length; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf || IS_NORM_0(leaf->scale)) {
                return;
            }
        }

        if (!leaf || IS_NORM_0(leaf->scale)) {
            return;
        }

        if (!endNode) {
            const std::lock_guard<std::mutex> destLock(destMutex);
            // Now that we've locked, is the endNode still not set?
            if (!endNode) {
                endNode = leaf->ShallowClone();
            }
        }

        leaf->branches[0] = NULL;
        leaf->branches[1] = NULL;
    });
    
    startNode->Prune(length);

    par_for(0, maxI, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < start; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf || IS_NORM_0(leaf->scale)) {
                return;
            }
        }
        leaf->branches[0] = endNode->branches[0];
        leaf->branches[1] = endNode->branches[1];
    });

    SetQubitCount(qubitCount - length);
    
    root->Prune(qubitCount);
}

real1_f QBinaryDecisionTree::Prob(bitLenInt qubit)
{
    Finish();

    bitCapInt qPower = pow2(qubit);
    bitCapInt maxI = qPower << ONE_BCI;

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    par_for(qPower, maxI, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = root->scale;
        for (bitLenInt j = 0; j <= qubit; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            if (!leaf) {
                break;
            }
            scale *= leaf->scale;
            if (IS_NORM_0(scale)) {
                return;
            }
        }
        oneChanceBuff[cpu] += norm(scale);
    });

    real1 oneChance = ZERO_R1;
    for (int i = 0; i < numCores; i++) {
        oneChance += oneChanceBuff[i];
    }

    return clampProb(oneChance);
}

real1_f QBinaryDecisionTree::ProbAll(bitCapInt fullRegister)
{
    Finish();

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
        if (IS_NORM_0(scale)) {
            break;
        }
    }

    return clampProb(norm(scale));
}

bool QBinaryDecisionTree::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    Finish();

    if (!doForce) {
        real1_f oneChance = Prob(qubit);
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            result = (Rand() <= oneChance);
        }
    }

    root->Branch(qubit + 1U);

    bitCapInt qPower = pow2(qubit);
    complex nrm = GetNonunitaryPhase();

    bitLenInt j;
    size_t bit;
    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < qPower; i++) {
        leaf = root;
        for (j = 0; j < qubit; j++) {
            bit = (i >> j) & 1U;
            leaf = leaf->branches[bit];
            if (!leaf || IS_NORM_0(leaf->scale)) {
                break;
            }
        }

        if (!leaf || IS_NORM_0(leaf->scale)) {
            continue;
        }

        if (result) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale = nrm;
        } else {
            leaf->branches[0]->scale = nrm;
            leaf->branches[1]->SetZero();
        }
    }

    root->Prune(qubit + 1U);

    return result;
}

bitCapInt QBinaryDecisionTree::MAll()
{
    Finish();

    bitCapInt result = 0;
    bool bitResult;
    real1_f oneChance;
    QBinaryDecisionTreeNodePtr leaf = root;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        leaf->Branch();
        oneChance = clampProb(norm(leaf->branches[1]->scale));
        if (oneChance >= ONE_R1) {
            bitResult = true;
        } else if (oneChance <= ZERO_R1) {
            bitResult = false;
        } else {
            bitResult = (Rand() <= oneChance);
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

void QBinaryDecisionTree::Apply2x2OnLeaf(const complex* mtrx, QBinaryDecisionTreeNodePtr leaf)
{
    QBinaryDecisionTreeNodePtr& branch0 = leaf->branches[0];
    QBinaryDecisionTreeNodePtr& branch1 = leaf->branches[1];

    bool wasLeaf0Norm0 = IS_NORM_0(branch0->scale);
    bool wasLeaf1Norm0 = IS_NORM_0(branch1->scale);

    // Apply gate.
    complex Y0 = branch0->scale;
    branch0->scale = mtrx[0] * Y0 + mtrx[1] * branch1->scale;
    branch1->scale = mtrx[2] * Y0 + mtrx[3] * branch1->scale;

    bool isLeaf0Norm0 = IS_NORM_0(branch0->scale);
    bool isLeaf1Norm0 = IS_NORM_0(branch1->scale);

    if (wasLeaf0Norm0 && !isLeaf0Norm0) {
        branch0->branches[0] = branch1->branches[0];
        branch0->branches[1] = branch1->branches[1];
    }
    if (wasLeaf1Norm0 && !isLeaf1Norm0) {
        branch1->branches[0] = branch0->branches[0];
        branch1->branches[1] = branch0->branches[1];
    }

    if (isLeaf0Norm0) {
        branch0->SetZero();
    }
    if (isLeaf1Norm0) {
        branch1->SetZero();
    }

    Y0 = sqrt(norm(branch0->scale) + norm(branch1->scale));
    branch0->scale /= Y0;
    branch1->scale /= Y0;
}

void QBinaryDecisionTree::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    bitCapInt targetPow = pow2(target);
    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    Dispatch(targetPow, [this, mtrx, target, targetPow]() {
        root->Branch(target + 1U);

        par_for(0, targetPow, [&](const bitCapInt& i, const int& cpu) {
            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                leaf = leaf->branches[(i >> j) & 1U];
                if (!leaf || IS_NORM_0(leaf->scale)) {
                    return;
                }
            }

            Apply2x2OnLeaf(mtrx.get(), leaf);
        });

        root->Prune(target + 1U);
    });
}

void QBinaryDecisionTree::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* lMtrx)
{
    if (!controlLen) {
        ApplySingleBit(lMtrx, target);
        return;
    }

    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    std::shared_ptr<bitCapInt[]> qPowersSorted(new bitCapInt[controlLen]);
    bitCapInt lowControlMask = 0;
    for (bitLenInt c = 0; c < controlLen; c++) {
        qPowersSorted[c] = pow2(sortedControls[c]);
        lowControlMask |= qPowersSorted[c];
    }
    // TODO: This is a horrible kludge.
    if (target < sortedControls.back()) {
        // At least one control bit index is higher than the target.
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->ApplyControlledSingleBit(controls, controlLen, target, lMtrx); });
        return;
    }

    bitCapInt targetPow = pow2(target);

    Dispatch(targetPow, [this, mtrx, target, targetPow, lowControlMask, qPowersSorted, controlLen]() {
        root->Branch(target + 1U);

        par_for_mask(0, targetPow, qPowersSorted.get(), controlLen, [&](const bitCapInt& lcv, const int& cpu) {
            // If any controls aren't set, skip.
            bitCapInt i = lcv | lowControlMask;

            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                leaf = leaf->branches[(i >> j) & 1U];
                if (!leaf || IS_NORM_0(leaf->scale)) {
                    return;
                }
            }

            Apply2x2OnLeaf(mtrx.get(), leaf);
        });

        root->Prune(target + 1U);
    });
}

} // namespace Qrack
