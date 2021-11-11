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
    , bdtThreshold(30)
    , maxQPowerOcl(pow2Ocl(qBitCount))
    , isFusionFlush(false)
    , shards(qBitCount)
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

#if ENABLE_OPENCL
    if ((engines.size() == 1U) && (engines[0] == QINTERFACE_QPAGER) && !OCLEngine::Instance()->GetDeviceCount()) {
        engines[0] = QINTERFACE_CPU;
    }
#endif

    if (getenv("QRACK_BDT_THRESHOLD")) {
        bdtThreshold = (bitLenInt)std::stoi(std::string(getenv("QRACK_BDT_THRESHOLD")));
    }

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

    return toRet;
}

void QBinaryDecisionTree::SetPermutation(bitCapInt initState, complex phaseFac)
{
    DumpBuffers();
    Dump();

    if (qubitCount <= bdtThreshold) {
        root = NULL;
        if (stateVecUnit == NULL) {
            stateVecUnit = MakeStateVector();
        }
        stateVecUnit->SetPermutation(initState, phaseFac);
        return;
    }

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
    FlushBuffers();
    Finish();

    QBinaryDecisionTreePtr copyPtr =
        std::make_shared<QBinaryDecisionTree>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
            false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);

    copyPtr->root = root ? root->ShallowClone() : NULL;
    copyPtr->stateVecUnit = stateVecUnit ? stateVecUnit->Clone() : NULL;

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
    root = std::make_shared<QBinaryDecisionTreeNode>();

    par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }
        setLambda((bitCapIntOcl)i, leaf);
    });

    root->ConvertStateVector(qubitCount);
    root->Prune(qubitCount);
}
void QBinaryDecisionTree::GetQuantumState(complex* state)
{
    if (stateVecUnit) {
        stateVecUnit->GetQuantumState(state);
        return;
    }
    FlushBuffers();
    GetTraversal([state](bitCapIntOcl i, complex scale) { state[i] = scale; });
}
void QBinaryDecisionTree::GetQuantumState(QInterfacePtr eng)
{
    GetTraversal([eng](bitCapIntOcl i, complex scale) { eng->SetAmplitude(i, scale); });
}
void QBinaryDecisionTree::SetQuantumState(const complex* state)
{
    if (stateVecUnit) {
        stateVecUnit->SetQuantumState(state);
        return;
    }
    DumpBuffers();
    Dump();
    SetTraversal([state](bitCapIntOcl i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = state[i]; });
}
void QBinaryDecisionTree::SetQuantumState(QInterfacePtr eng)
{
    Finish();
    SetTraversal([eng](bitCapIntOcl i, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = eng->GetAmplitude(i); });
}
void QBinaryDecisionTree::GetProbs(real1* outputProbs)
{
    if (stateVecUnit) {
        stateVecUnit->GetProbs(outputProbs);
        return;
    }
    FlushBuffers();
    GetTraversal([outputProbs](bitCapIntOcl i, complex scale) { outputProbs[i] = norm(scale); });
}

real1_f QBinaryDecisionTree::SumSqrDiff(QBinaryDecisionTreePtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        toCompare->SetStateVector();
        return stateVecUnit->SumSqrDiff(toCompare->stateVecUnit);
    }

    ResetStateVector();
    FlushBuffers();
    Finish();
    toCompare->ResetStateVector();
    toCompare->FlushBuffers();
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
    if (stateVecUnit) {
        return stateVecUnit->GetAmplitude(perm);
    }

    FlushBuffers();
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
    if ((qubitCount + toCopy->qubitCount) <= bdtThreshold) {
        SetStateVector();
        toCopy->SetStateVector();
        shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return stateVecUnit->Compose(toCopy->stateVecUnit, start);
    }

    if (start && (start != qubitCount)) {
        return QInterface::Compose(toCopy, start);
    }

    ResetStateVector();
    Finish();
    toCopy->ResetStateVector();
    toCopy->Finish();

    bitLenInt qbCount;
    bitCapIntOcl maxI;

    QBinaryDecisionTreeNodePtr rootClone = toCopy->root->ShallowClone();
    if (start) {
        qbCount = qubitCount;
        maxI = maxQPowerOcl;
    } else {
        qbCount = toCopy->qubitCount;
        maxI = toCopy->maxQPowerOcl;
        root.swap(rootClone);
    }

    par_for_qbdt(0, maxI, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qbCount; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return pow2Ocl(qbCount - j) - ONE_BCI;
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
void QBinaryDecisionTree::DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    if (stateVecUnit && ((qubitCount - length) <= bdtThreshold)) {
        if (dest) {
            dest->SetStateVector();
            stateVecUnit->Decompose(start, dest->stateVecUnit);
            std::copy(shards.begin() + start, shards.begin() + start + length, dest->shards.begin());
        } else {
            stateVecUnit->Dispose(start, length);
        }
        shards.erase(shards.begin() + start, shards.begin() + start + length);
        SetQubitCount(qubitCount - length);

        return;
    }

    bitLenInt end = start + length;

    if (start && (end < qubitCount)) {
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

    ResetStateVector();
    Finish();
    if (dest) {
        dest->ResetStateVector();
        dest->DumpBuffers();
        dest->Dump();
    }

    bool isReversed = !start;
    if (isReversed) {
        start = length;
        length = qubitCount - length;
    }

    bitLenInt j;
    bitCapIntOcl maxI = pow2Ocl(start);
    QBinaryDecisionTreeNodePtr startNode = NULL;
    par_for_qbdt(0, maxI, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (j = 0; j < start; j++) {
            if (IS_NORM_0(leaf->scale)) {
                // WARNING: Mutates loop control variable!
                return pow2Ocl(start - j) - ONE_BCI;
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

real1_f QBinaryDecisionTree::Prob(bitLenInt qubit)
{
    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        return stateVecUnit->Prob(qubit);
    }

    ResetStateVector();
    FlushBuffer(qubit);
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
    if (stateVecUnit) {
        return stateVecUnit->ProbAll(fullRegister);
    }

    FlushBuffers();
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
    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        return stateVecUnit->ForceM(qubit, result, doForce, doApply);
    }

    if (doForce) {
        if (doApply) {
            ExecuteAsStateVector([&](QInterfacePtr eng) { eng->ForceM(qubit, result, true, doApply); });
        }
        return result;
    }

    real1_f oneChance = Prob(qubit);
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

    ResetStateVector();
    FlushBuffer(qubit);
    Finish();

    bitCapIntOcl qPower = pow2Ocl(qubit);
    root->scale = GetNonunitaryPhase();

    par_for(0, qPower, [&](const bitCapInt i, const int cpu) {
        QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt j = 0; j < qubit; j++) {
            if (IS_NORM_0(leaf->scale)) {
                return;
            }
            leaf->Branch();
            leaf = leaf->branches[SelectBit(i, j)];
        }

        if (IS_NORM_0(leaf->scale)) {
            return;
        }
        leaf->Branch();

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
    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        return stateVecUnit->MAll();
    }

    ResetStateVector();
    FlushBuffers();
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
    bitCapInt highControlMask, bool isAnti, bool isParallel)
{
    leaf->Branch();
    bitLenInt remainder = qubitCount - (depth + 1);

    QBinaryDecisionTreeNodePtr& b0 = leaf->branches[0];
    QBinaryDecisionTreeNodePtr& b1 = leaf->branches[1];

    bitCapIntOcl remainderPow = pow2Ocl(remainder);
    bitCapIntOcl maskTarget = (isAnti ? 0U : highControlMask);

    IncrementFunc fn = [&](const bitCapInt i, const int cpu) {
        size_t bit;
        bool isZero;

        QBinaryDecisionTreeNodePtr leaf0 = b0;
        QBinaryDecisionTreeNodePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        // b0 and b1 can't both be 0.
        isZero = false;

        bitLenInt j = 0;
        for (j = 0; j < remainder; j++) {
            leaf0->Branch(1, true);
            leaf1->Branch(1, true);

            bit = SelectBit(i, remainder - (j + 1U));

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
            return pow2Ocl(remainder - (j + 1U)) - ONE_BCI;
        }

        if ((i & highControlMask) != maskTarget) {
            leaf0->scale = scale0;
            leaf1->scale = scale1;

            return (bitCapIntOcl)0U;
        }

        complex Y0 = scale0;
        complex Y1 = scale1;
        leaf0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        leaf1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return (bitCapIntOcl)0U;
    };

    if (isParallel) {
        par_for_qbdt(0, remainderPow, fn);
    } else {
        for (bitCapIntOcl i = 0; i < remainderPow; i++) {
            i |= fn(i, 0);
        }
    }

    b0->ConvertStateVector(remainder);
    b1->ConvertStateVector(remainder);
    leaf->Prune(remainder + 1U);
}

template <typename Fn> void QBinaryDecisionTree::ApplySingle(const complex* lMtrx, bitLenInt target, Fn leafFunc)
{
    std::shared_ptr<complex> mtrx(new complex[4], std::default_delete<complex[]>());
    std::copy(lMtrx, lMtrx + 4U, mtrx.get());
    bitCapIntOcl targetPow = pow2Ocl(target);

    ResetStateVector();

    Dispatch(targetPow, [this, mtrx, target, targetPow, leafFunc]() {
        bool isParallel = (pow2Ocl(target) < GetParallelThreshold());

        par_for_qbdt(0, targetPow, [&](const bitCapInt i, const int cpu) {
            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    return pow2Ocl(target - j) - ONE_BCI;
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

void QBinaryDecisionTree::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4];
    if (shards[target]) {
        shards[target]->Compose(lMtrx);
        std::copy(shards[target]->gate, shards[target]->gate + 4, mtrx);
        shards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }
    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }

    if (!isFusionFlush) {
        if (stateVecUnit && (qubitCount <= bdtThreshold)) {
            stateVecUnit->ApplySingleBit(mtrx, target);
            return;
        }
        ResetStateVector();
        shards[target] = std::make_shared<MpsShard>(mtrx);
        return;
    }

    ApplySingle(mtrx, target,
        [this, target](QBinaryDecisionTreeNodePtr leaf, const complex* mtrx, bitCapInt ignored, bool isParallel) {
            Apply2x2OnLeaf(mtrx, leaf, target, 0U, false, isParallel);
        });
}

void QBinaryDecisionTree::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    if (shards[target]) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (IS_NORM_0(topLeft - bottomRight) && (randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft))) {
        return;
    }

    ApplySingle(
        mtrx, target, [](QBinaryDecisionTreeNodePtr leaf, const complex* mtrx, bitCapInt ignored, bool ignored2) {
            leaf->Branch();
            leaf->branches[0]->scale *= mtrx[0];
            leaf->branches[1]->scale *= mtrx[3];
            leaf->Prune();
        });
}

void QBinaryDecisionTree::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    if (shards[target]) {
        ApplySingleBit(mtrx, target);
        return;
    }

    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        stateVecUnit->ApplySingleInvert(topRight, bottomLeft, target);
        return;
    }

    ApplySingle(
        mtrx, target, [](QBinaryDecisionTreeNodePtr leaf, const complex* mtrx, bitCapInt ignored, bool ignored2) {
            leaf->Branch();
            leaf->branches[0].swap(leaf->branches[1]);
            leaf->branches[0]->scale *= mtrx[1];
            leaf->branches[1]->scale *= mtrx[2];
            leaf->Prune();
        });
}

template <typename Lfn>
void QBinaryDecisionTree::ApplyControlledSingle(const complex* lMtrx, const bitLenInt* controls,
    const bitLenInt& controlLen, const bitLenInt& target, bool isAnti, Lfn leafFunc)
{
    if (!controlLen) {
        ApplySingle(lMtrx, target, leafFunc);
        return;
    }

    std::shared_ptr<complex> mtrxS(new complex[4], std::default_delete<complex[]>());
    std::copy(lMtrx, lMtrx + 4, mtrxS.get());

    std::vector<bitLenInt> sortedControls(controlLen);
    std::copy(controls, controls + controlLen, sortedControls.begin());
    std::sort(sortedControls.begin(), sortedControls.end());

    std::vector<bitCapIntOcl> qPowersSorted;
    bitCapIntOcl lowControlMask = 0U;
    bitLenInt c;
    for (c = 0U; (c < controlLen) && (sortedControls[c] < target); c++) {
        qPowersSorted.push_back(pow2Ocl(target - (sortedControls[c] + ONE_BCI)));
        lowControlMask |= qPowersSorted.back();
    }
    std::reverse(qPowersSorted.begin(), qPowersSorted.end());

    bitCapIntOcl highControlMask = 0U;
    for (; c < controlLen; c++) {
        highControlMask |= pow2Ocl(qubitCount - (sortedControls[c] + ONE_BCI));
    }

    bitCapIntOcl targetPow = pow2Ocl(target);
    bitCapIntOcl maskTarget = (isAnti ? 0U : lowControlMask);

    ResetStateVector();

    FlushBuffer(target);
    for (bitLenInt i = 0U; i < controlLen; i++) {
        FlushBuffer(controls[i]);
    }

    Dispatch(targetPow, [this, mtrxS, target, targetPow, qPowersSorted, highControlMask, maskTarget, leafFunc]() {
        complex* mtrx = mtrxS.get();

        if (qPowersSorted.size()) {
            root->Branch(target);
        }

        bool isPhase = false;
        bool isInvert = false;
        if (!highControlMask) {
            isPhase = ((mtrx[1] == ZERO_CMPLX) && (mtrx[2] == ZERO_CMPLX));
            isInvert = ((mtrx[0] == ZERO_CMPLX) && (mtrx[3] == ZERO_CMPLX));
        }

        bitCapInt maxLcv = targetPow >> qPowersSorted.size();
        bool isParallel = (maxLcv < GetParallelThreshold());

        par_for_qbdt(0, maxLcv, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt i = 0U;
            bitCapInt iHigh = lcv;
            bitCapInt iLow;
            int p;
            for (p = 0; p < (int)qPowersSorted.size(); p++) {
                iLow = iHigh & (qPowersSorted[p] - ONE_BCI);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << ONE_BCI;
            }
            i |= iHigh | maskTarget;

            QBinaryDecisionTreeNodePtr leaf = root;
            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                if (IS_NORM_0(leaf->scale)) {
                    // WARNING: Mutates loop control variable!
                    i = pow2Ocl(target - j) - ONE_BCI;
                    for (p = (int)(qPowersSorted.size() - 1U); p >= 0; p--) {
                        i = RemovePower(i, qPowersSorted[p]);
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
    });
}

void QBinaryDecisionTree::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        stateVecUnit->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    ApplyControlledSingle(mtrx, controls, controlLen, target, false,
        [this, target](QBinaryDecisionTreeNodePtr leaf, const complex* mtrx, bitCapInt highControlMask,
            bool isParallel) { Apply2x2OnLeaf(mtrx, leaf, target, highControlMask, false, isParallel); });
}

void QBinaryDecisionTree::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (qubitCount <= bdtThreshold) {
        SetStateVector();
        stateVecUnit->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
        return;
    }

    ApplyControlledSingle(mtrx, controls, controlLen, target, true,
        [this, target](QBinaryDecisionTreeNodePtr leaf, const complex* mtrx, bitCapInt highControlMask,
            bool isParallel) { Apply2x2OnLeaf(mtrx, leaf, target, highControlMask, true, isParallel); });
}

} // namespace Qrack
