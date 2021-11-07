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
#include "qfactory.hpp"

#include <mutex>

namespace Qrack {

QBinaryDecisionTree::QBinaryDecisionTree(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> ignored,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1)
    , engines(eng)
    , devID(deviceId)
    , root(NULL)
    , stateVecUnit(NULL)
    , maxQPowerOcl(pow2Ocl(qBitCount))
    , zeroMasks(qubitCount)
{
    if ((engines[0] == QINTERFACE_HYBRID) || (engines[0] == QINTERFACE_OPENCL)) {
#if ENABLE_OPENCL
        if (!OCLEngine::Instance()->GetDeviceCount()) {
            engines[0] = QINTERFACE_CPU;
        }
#else
        engines[0] = QINTERFACE_CPU;
#endif
    }

    pStridePow =
        getenv("QRACK_PSTRIDEPOW") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW;
    SetConcurrency(std::thread::hardware_concurrency());
    SetPermutation(initState);
}

QInterfacePtr QBinaryDecisionTree::MakeStateVector()
{
    return CreateQuantumInterface(engines, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
        false, devID, hardware_rand_generator != NULL, false, amplitudeFloor);
}

bool QBinaryDecisionTree::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    SetStateVector();
    bool toRet = stateVecUnit->ForceMParity(mask, result, doForce);
    ResetStateVector();

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
        bit = SelectBit(initState, qubit);
        leaf->branches[bit] = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
        leaf->branches[bit ^ 1U] = std::make_shared<QBinaryDecisionTreeNode>(ZERO_CMPLX);
        leaf = leaf->branches[bit];
    }
}

QInterfacePtr QBinaryDecisionTree::Clone()
{
    Finish();

    QBinaryDecisionTreePtr copyPtr =
        std::make_shared<QBinaryDecisionTree>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
            false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root->ShallowClone();

    return copyPtr;
}

template <typename Fn> void QBinaryDecisionTree::GetTraversal(Fn getLambda)
{
    Finish();

    par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = leaf->scale;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale)) {
                break;
            }
            leaf = leaf->branches[SelectBit(i, j)];
            scale *= leaf->scale;
        }
        getLambda((bitCapIntOcl)i, scale);
    });
}
template <typename Fn> void QBinaryDecisionTree::SetTraversal(Fn setLambda)
{
    Dump();

    root = std::make_shared<QBinaryDecisionTreeNode>();
    root->Branch(qubitCount);

    par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[SelectBit(i, j)];
        }
        setLambda((bitCapIntOcl)i, leaf);
    });

    root->ConvertStateVector(qubitCount);
    root->Prune(qubitCount);
}
void QBinaryDecisionTree::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapIntOcl i, complex scale) { state[i] = scale; });
}
void QBinaryDecisionTree::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapIntOcl i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBinaryDecisionTree::SetQuantumState(const complex* state)
{
    SetTraversal([state](bitCapIntOcl i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = state[i]; });
}
void QBinaryDecisionTree::SetQuantumState(QInterfacePtr eng)
{
    SetTraversal([eng](bitCapIntOcl i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = eng->GetAmplitude(i); });
}
void QBinaryDecisionTree::GetProbs(real1* outputProbs)
{
    GetTraversal([outputProbs](bitCapIntOcl i, complex scale) { outputProbs[i] = norm(scale); });
}

real1_f QBinaryDecisionTree::SumSqrDiff(QBinaryDecisionTreePtr toCompare)
{
    Finish();
    toCompare->Finish();

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> partInner(new complex[numCores]());

    par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf1 = root;
        QBinaryDecisionTreeNodePtr leaf2 = toCompare->root;
        complex scale1 = leaf1->scale;
        complex scale2 = leaf2->scale;
        bitLenInt j;
        for (j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale1)) {
                return;
            }
            leaf1 = leaf1->branches[SelectBit(i, j)];
            scale1 *= leaf1->scale;
        }
        for (j = 0; j < qubitCount; j++) {
            if (IS_NORM_0(scale2)) {
                return;
            }
            leaf2 = leaf2->branches[SelectBit(i, j)];
            scale2 *= leaf2->scale;
        }
        partInner[cpu] += conj(scale2) * scale1;
    });

    complex projection = ZERO_CMPLX;
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
        leaf = leaf->branches[SelectBit(perm, j)];
        scale *= leaf->scale;
    }

    return scale;
}

bitLenInt QBinaryDecisionTree::Compose(QBinaryDecisionTreePtr toCopy, bitLenInt start)
{
    if (start != qubitCount) {
        return QInterface::Compose(toCopy, start);
    }

    Finish();
    toCopy->Finish();

    bitLenInt qbCount;
    bitCapIntOcl maxI;
    qbCount = qubitCount;
    maxI = maxQPowerOcl;

    par_for(0, maxI, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qbCount; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (IS_NORM_0(leaf->scale)) {
            return;
        }

        leaf->branches[0] = toCopy->root->branches[0];
        leaf->branches[1] = toCopy->root->branches[1];
    });

    zeroMasks.insert(zeroMasks.end(), toCopy->zeroMasks.begin(), toCopy->zeroMasks.end());
    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}
void QBinaryDecisionTree::DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    bitLenInt end = start + length;

    if (end != qubitCount) {
        bitLenInt offset = qubitCount - end;

        ROL(offset, 0, qubitCount);

        if (dest) {
            Decompose(qubitCount - length, dest);
        } else {
            Dispose(qubitCount - length, length);
        }

        ROR(offset, 0, qubitCount);

        return;
    }

    Finish();
    if (dest) {
        dest->Dump();
    }

    bitCapIntOcl maxI = pow2Ocl(start);
    QBinaryDecisionTreeNodePtr startNode = NULL;
    std::mutex destMutex;
    par_for(0, maxI, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < start; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[SelectBit(i, j)];
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

    startNode->scale /= abs(startNode->scale);

    if (dest) {
        dest->root = startNode;
        std::copy(zeroMasks.begin() + start, zeroMasks.end(), dest->zeroMasks.begin());
    }

    zeroMasks.erase(zeroMasks.begin() + start, zeroMasks.end());
    SetQubitCount(qubitCount - length);
}

real1_f QBinaryDecisionTree::Prob(bitLenInt qubit)
{
    Finish();

    bitCapIntOcl qPower = pow2Ocl(qubit);

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    par_for(0, qPower, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = root->scale;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(scale)) {
                return;
            }
            leaf = leaf->branches[SelectBit(i, j)];
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
        leaf = leaf->branches[SelectBit(fullRegister, j)];
        scale *= leaf->scale;
    }

    return clampProb(norm(scale));
}

bool QBinaryDecisionTree::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (doForce) {
        if (doApply) {
            ExecuteAsStateVector([&](QInterfacePtr eng) { eng->ForceM(qubit, result, true, doApply); });
        }
        return result;
    }

    Finish();

    real1_f oneChance = Prob(qubit);
    if (oneChance >= ONE_R1) {
        result = true;
    } else if (oneChance <= ZERO_R1) {
        result = false;
    } else {
        result = (Rand() <= oneChance);
    }

    root->Branch(qubit + 1U);

    bitCapIntOcl qPower = pow2Ocl(qubit);
    root->scale = GetNonunitaryPhase();

    par_for(0, qPower, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf = leaf->branches[SelectBit(i, j)];
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

void QBinaryDecisionTree::Apply2x2OnLeaf(const complex* mtrx, QBinaryDecisionTreeNodePtr leaf, bitLenInt depth,
    bool isParallel, bitCapInt highControlMask, bool isAnti)
{
    leaf->Branch();
    bitLenInt remainder = qubitCount - (depth + 1);

    QBinaryDecisionTreeNodePtr& b0 = leaf->branches[0];
    QBinaryDecisionTreeNodePtr& b1 = leaf->branches[1];

    bitCapIntOcl remainderPow = pow2Ocl(remainder);
    std::vector<std::set<bitCapInt>> halfZeroMasks = zeroMasks;
    bitLenInt j, jAlign;
    size_t bit;
    bool isZero;

    for (bitCapIntOcl i = 0; i < remainderPow; i++) {
        for (j = 0; j < remainder; j++) {
            jAlign = j + depth + 1U;
            if (halfZeroMasks[jAlign].find(i & (pow2Ocl(j + 1U) - ONE_BCI)) != halfZeroMasks[jAlign].end()) {
                break;
            }
        }

        if (j < remainder) {
            continue;
        }

        QBinaryDecisionTreeNodePtr leaf0 = b0;
        QBinaryDecisionTreeNodePtr leaf1 = b1;

        QBinaryDecisionTreeNodePtr prevLeaf0, prevLeaf1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        // b0 and b1 can't both be 0.
        isZero = false;

        for (j = 0; j < remainder; j++) {
            leaf0->Branch(1, true);
            leaf1->Branch(1, true);

            bit = SelectBit(i, j);

            prevLeaf0 = leaf0;
            leaf0 = leaf0->branches[bit];
            scale0 *= leaf0->scale;

            prevLeaf1 = leaf1;
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

            bitCapInt mask = (i & (pow2Ocl(j + 1U) - ONE_BCI));

            jAlign = j + depth + 1U;

            halfZeroMasks[jAlign].insert(mask);

            bit ^= 1U;
            leaf0 = prevLeaf0->branches[bit];
            leaf1 = prevLeaf1->branches[bit];

            if (IS_NORM_0(leaf0->scale) && IS_NORM_0(leaf1->scale)) {
                zeroMasks[jAlign].insert(mask);
            }

            continue;
        }

        if ((i & highControlMask) != (isAnti ? 0U : highControlMask)) {
            leaf0->scale = scale0;
            leaf1->scale = scale1;
            continue;
        }

        complex Y0 = scale0;
        complex Y1 = scale1;
        leaf0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        leaf1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;
    }

    b0->ConvertStateVector(remainder);
    b1->ConvertStateVector(remainder);
    leaf->Prune(remainder + 1U);
}

template <typename Fn> void QBinaryDecisionTree::ApplySingle(bitLenInt target, Fn leafFunc)
{
    bitCapIntOcl targetPow = pow2Ocl(target);
    Dispatch(targetPow, [this, target, targetPow, leafFunc]() {
        root->Branch(target);

        zeroMasks[target].clear();

        bool isParallel = (targetPow < GetParallelThreshold());
        for (bitCapInt i = 0; i < targetPow; i++) {
            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    break;
                }
                leaf = leaf->branches[SelectBit(i, j)];
            }

            if (IS_NORM_0(leaf->scale)) {
                continue;
            }

            leafFunc(leaf, isParallel, 0U);
        }

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
        Apply2x2OnLeaf(mtrx.get(), leaf, target, isParallel, 0U, false);
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
        leaf->branches[0].swap(leaf->branches[1]);
        leaf->branches[0]->scale *= topRight;
        leaf->branches[1]->scale *= bottomLeft;
        leaf->Prune();
    });
}

template <typename Lfn>
void QBinaryDecisionTree::ApplyControlledSingle(bool isAnti, std::shared_ptr<complex[]> mtrx, const bitLenInt* controls,
    const bitLenInt& controlLen, const bitLenInt& target, Lfn leafFunc)
{
    if (!controlLen) {
        ApplySingle(target, leafFunc);
        return;
    }

    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    std::vector<bitCapIntOcl> qPowersSorted(controlLen);
    bitCapIntOcl lowControlMask = 0U;
    bitLenInt c;
    for (c = 0U; (c < controlLen) && (sortedControls[c] < target); c++) {
        qPowersSorted[c] = pow2Ocl(sortedControls[c]);
        lowControlMask |= qPowersSorted[c];
    }
    bitLenInt controlBound = c;
    bitCapIntOcl highControlMask = 0U;
    for (; c < controlLen; c++) {
        qPowersSorted[c] = pow2Ocl(sortedControls[c]);
        highControlMask |= qPowersSorted[c];
    }
    highControlMask >>= (target + 1U);

    bitCapIntOcl targetPow = pow2Ocl(target);

    Dispatch(targetPow,
        [this, isAnti, mtrx, target, targetPow, lowControlMask, highControlMask, qPowersSorted, controlBound,
            leafFunc]() {
            root->Branch(target);

            zeroMasks[target].clear();

            bool isPhase = false;
            bool isInvert = false;
            if (!highControlMask) {
                isPhase = ((mtrx[1] == ZERO_CMPLX) && (mtrx[2] == ZERO_CMPLX));
                isInvert = ((mtrx[0] == ZERO_CMPLX) && (mtrx[3] == ZERO_CMPLX));
            }
            bool isParallel = ((targetPow >> controlBound) < GetParallelThreshold());

            bitCapInt i, iHigh, iLow;
            bitLenInt p;
            bitCapInt maxLcv = targetPow >> controlBound;
            for (bitCapInt lcv = 0U; lcv < maxLcv; lcv++) {
                iHigh = lcv;
                i = 0U;
                for (p = 0U; p < controlBound; p++) {
                    iLow = iHigh & (qPowersSorted[p] - ONE_BCI);
                    i |= iLow;
                    iHigh = (iHigh ^ iLow) << ONE_BCI;
                }
                i |= iHigh;

                // If any controls aren't set, skip, (but we push apart).
                if (!isAnti) {
                    i |= lowControlMask;
                }

                QBinaryDecisionTreeNodePtr leaf = root;
                // Iterate to qubit depth.
                for (bitLenInt j = 0; j < target; j++) {
                    if (IS_NORM_0(leaf->scale)) {
                        break;
                    }
                    leaf = leaf->branches[SelectBit(i, j)];
                }

                if (IS_NORM_0(leaf->scale)) {
                    continue;
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
                    leafFunc(leaf, isParallel, highControlMask);
                }
            }

            root->Prune(target);
        });
}

void QBinaryDecisionTree::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* lMtrx)
{
    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    ApplyControlledSingle(false, mtrx, controls, controlLen, target,
        [this, mtrx, target](QBinaryDecisionTreeNodePtr leaf, bool isParallel, bitCapInt highControlMask) {
            Apply2x2OnLeaf(mtrx.get(), leaf, target, isParallel, highControlMask, false);
        });
}

void QBinaryDecisionTree::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* lMtrx)
{
    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    ApplyControlledSingle(true, mtrx, controls, controlLen, target,
        [this, mtrx, target](QBinaryDecisionTreeNodePtr leaf, bool isParallel, bitCapInt highControlMask) {
            Apply2x2OnLeaf(mtrx.get(), leaf, target, isParallel, highControlMask, true);
        });
}

} // namespace Qrack
