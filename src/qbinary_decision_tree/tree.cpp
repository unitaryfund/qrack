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
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[(i >> j) & 1U];
            scale *= leaf->scale;
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
            if (IS_NORM_0(scale1)) {
                return;
            }
            leaf1 = leaf1->branches[(i >> j) & 1U];
            scale1 *= leaf1->scale;
        }
        for (j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale2)) {
                return;
            }
            leaf2 = leaf2->branches[(i >> j) & 1U];
            scale2 *= leaf2->scale;
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
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[(perm >> j) & 1U];
        scale *= leaf->scale;
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
                if (IS_NORM_0(leaf->scale)) {
                    return;
                }
                leaf = leaf->branches[(i >> j) & 1U];
            }

            if (IS_NORM_0(leaf->scale)) {
                return;
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
                if (IS_NORM_0(leaf->scale)) {
                    return;
                }
                leaf = leaf->branches[(i >> j) & 1U];
            }

            if (IS_NORM_0(leaf->scale)) {
                return;
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
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[(i >> j) & 1U];
        }

        if (IS_NORM_0(leaf->scale)) {
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
        SetQubitCount(qubitCount - length);
        root->Prune(qubitCount);
        if (dest) {
            dest->root->Prune(length);
        }
        return;
    }

    bitCapInt lengthPow = pow2(length);
    QBinaryDecisionTreeNodePtr endNode = NULL;

    par_for(0, lengthPow, [&](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf = startNode;
        for (bitLenInt j = 0; j < length; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[(i >> j) & 1U];
        }

        if (IS_NORM_0(leaf->scale)) {
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
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[(i >> j) & 1U];
        }

        if (IS_NORM_0(leaf->scale)) {
            return;
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

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    par_for(0, qPower, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = root->scale;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(scale)) {
                return;
            }
            leaf = leaf->branches[(i >> j) & 1U];
            scale *= leaf->scale;
        }

        if (IS_NORM_0(scale)) {
            return;
        }

        oneChanceBuff[cpu] += norm(scale * leaf->branches[1]->scale);
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
        if (IS_NORM_0(scale)) {
            break;
        }
        leaf = leaf->branches[(fullRegister >> j) & 1U];
        scale *= leaf->scale;
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
    root->scale = GetNonunitaryPhase();

    par_for(0, qPower, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[(i >> j) & 1U];
        }

        if (IS_NORM_0(leaf->scale)) {
            return;
        }

        if (result) {
            leaf->branches[0]->SetZero();
            leaf->branches[1]->scale /= abs(leaf->branches[1]->scale);
        } else {
            leaf->branches[0]->scale /= abs(leaf->branches[0]->scale);
            leaf->branches[1]->SetZero();
        }
    });

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

void QBinaryDecisionTree::Apply2x2OnLeaf(
    const complex* mtrx, QBinaryDecisionTreeNodePtr leaf, bitLenInt depth, bool isParallel, bitCapInt highControlMask)
{
    leaf->Branch(1, true);
    bitLenInt remainder = qubitCount - (depth + 1);

    QBinaryDecisionTreeNodePtr& b0 = leaf->branches[0];
    QBinaryDecisionTreeNodePtr& b1 = leaf->branches[1];

    ParallelFunc fn = [b0, b1, remainder, highControlMask, mtrx](const bitCapInt& i, const int& cpu) {
        QBinaryDecisionTreeNodePtr leaf0 = b0;
        QBinaryDecisionTreeNodePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        size_t bit;
        for (bitLenInt j = 0; j < remainder; j++) {
            leaf0->Branch(1, true);
            leaf1->Branch(1, true);

            bit = (i >> j) & 1U;

            leaf0 = leaf0->branches[bit];
            scale0 *= leaf0->scale;

            leaf1 = leaf1->branches[bit];
            scale1 *= leaf1->scale;
        }

        if ((i & highControlMask) != highControlMask) {
            leaf0->scale = scale0;
            leaf1->scale = scale1;
            return;
        }

        complex Y0 = scale0;
        complex Y1 = scale1;
        leaf0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        leaf1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;
    };

    bitCapInt remainderPow = pow2(remainder);
    if (isParallel) {
        par_for(0, remainderPow, fn);
    } else {
        for (bitCapInt i = 0; i < remainderPow; i++) {
            fn(i, 0);
        }
    }

    b0->ConvertStateVector(remainder);
    b1->ConvertStateVector(remainder);
    leaf->Prune(remainder + 1U);
}

template <typename Fn> void QBinaryDecisionTree::ApplySingle(bitLenInt target, Fn leafFunc)
{
    bitCapInt targetPow = pow2(target);
    Dispatch(targetPow, [this, target, targetPow, leafFunc]() {
        root->Branch(target);

        bool isParallel = (targetPow < GetParallelThreshold());
        par_for(0, targetPow, [&](const bitCapInt& i, const int& cpu) {
            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    return;
                }
                leaf = leaf->branches[(i >> j) & 1U];
            }

            if (IS_NORM_0(leaf->scale)) {
                return;
            }

            leafFunc(leaf, isParallel, 0U);
        });

        root->Prune(target);
    });
}

void QBinaryDecisionTree::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    if ((lMtrx[1] == ZERO_CMPLX) && (lMtrx[2] == ZERO_CMPLX)) {
        ApplySinglePhase(lMtrx[0], lMtrx[3], target);
        return;
    }
    if ((lMtrx[0] == ZERO_CMPLX) && (lMtrx[3] == ZERO_CMPLX)) {
        ApplySingleInvert(lMtrx[1], lMtrx[2], target);
        return;
    }

    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    ApplySingle(target, [this, mtrx, target](QBinaryDecisionTreeNodePtr leaf, bool isParallel, bitCapInt ignored) {
        Apply2x2OnLeaf(mtrx.get(), leaf, target, isParallel, 0U);
    });
}

void QBinaryDecisionTree::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    if ((topLeft == bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        return;
    }

    ApplySingle(target, [topLeft, bottomRight](QBinaryDecisionTreeNodePtr leaf, bool ignored1, bitCapInt ignored2) {
        leaf->Branch();
        leaf->branches[0]->scale *= topLeft;
        leaf->branches[1]->scale *= bottomRight;
        leaf->Prune();
    });
}

void QBinaryDecisionTree::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    ApplySingle(target, [topRight, bottomLeft](QBinaryDecisionTreeNodePtr leaf, bool ignored1, bitCapInt ignored2) {
        leaf->Branch();
        std::swap(leaf->branches[0], leaf->branches[1]);
        leaf->branches[0]->scale *= topRight;
        leaf->branches[1]->scale *= bottomLeft;
        leaf->Prune();
    });
}

template <typename Lfn>
void QBinaryDecisionTree::ApplyControlledSingle(std::shared_ptr<complex[]> mtrx, const bitLenInt* controls,
    const bitLenInt& controlLen, const bitLenInt& target, Lfn leafFunc)
{
    if (!controlLen) {
        ApplySingle(target, leafFunc);
        return;
    }

    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    std::shared_ptr<bitCapInt[]> qPowersSorted(new bitCapInt[controlLen]);
    bitCapInt lowControlMask = 0U;
    bitLenInt c;
    for (c = 0U; (c < controlLen) && (sortedControls[c] < target); c++) {
        qPowersSorted[c] = pow2(sortedControls[c]);
        lowControlMask |= qPowersSorted[c];
    }
    bitLenInt controlBound = c;
    bitCapInt highControlMask = 0U;
    for (; c < controlLen; c++) {
        qPowersSorted[c] = pow2(sortedControls[c]);
        highControlMask |= qPowersSorted[c];
    }
    highControlMask >>= (target + 1U);

    bitCapInt targetPow = pow2(target);

    Dispatch(targetPow,
        [this, mtrx, target, targetPow, lowControlMask, highControlMask, qPowersSorted, controlBound, leafFunc]() {
            root->Branch(target);

            bool isPhase = false;
            bool isInvert = false;
            if (!highControlMask) {
                isPhase = ((mtrx[1] == ZERO_CMPLX) && (mtrx[2] == ZERO_CMPLX));
                isInvert = ((mtrx[0] == ZERO_CMPLX) && (mtrx[3] == ZERO_CMPLX));
            }
            bool isParallel = ((targetPow >> controlBound) < GetParallelThreshold());
            par_for_mask(0, targetPow, qPowersSorted.get(), controlBound, [&](const bitCapInt& lcv, const int& cpu) {
                // If any controls aren't set, skip.
                bitCapInt i = lcv | lowControlMask;

                QBinaryDecisionTreeNodePtr leaf = root;
                // Iterate to qubit depth.
                for (bitLenInt j = 0; j < target; j++) {
                    if (IS_NORM_0(leaf->scale)) {
                        return;
                    }
                    leaf = leaf->branches[(i >> j) & 1U];
                }

                if (IS_NORM_0(leaf->scale)) {
                    return;
                }

                if (isPhase) {
                    leaf->Branch();
                    leaf->branches[0]->scale *= mtrx[0];
                    leaf->branches[1]->scale *= mtrx[3];
                    leaf->Prune();
                } else if (isInvert) {
                    leaf->Branch();
                    std::swap(leaf->branches[0], leaf->branches[1]);
                    leaf->branches[0]->scale *= mtrx[1];
                    leaf->branches[1]->scale *= mtrx[2];
                    leaf->Prune();
                } else {
                    leafFunc(leaf, isParallel, highControlMask);
                }
            });

            root->Prune(target);
        });
}

void QBinaryDecisionTree::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* lMtrx)
{
    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    ApplyControlledSingle(mtrx, controls, controlLen, target,
        [this, mtrx, target](QBinaryDecisionTreeNodePtr leaf, bool isParallel, bitCapInt highControlMask) {
            Apply2x2OnLeaf(mtrx.get(), leaf, target, isParallel, highControlMask);
        });
}

} // namespace Qrack
