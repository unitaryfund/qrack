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
    pStridePow =
        getenv("QRACK_PSTRIDEPOW") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW;
    SetConcurrency(std::thread::hardware_concurrency());
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

    copyPtr->root = root ? root->DeepClone() : NULL;

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
    complex scale = (complex)std::pow(SQRT1_2_R1, qubitCount);
    bitLenInt j;

    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
        }
        setLambda(i, scale, leaf);
    }

    root->Prune(qubitCount);
    root->Normalize(qubitCount);
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
    SetTraversal(
        [state](bitCapInt i, complex scale, QBinaryDecisionTreeNodePtr leaf) { leaf->scale = state[i] / scale; });
}
void QBinaryDecisionTree::SetQuantumState(QInterfacePtr eng)
{
    SetTraversal([eng](bitCapInt i, complex scale, QBinaryDecisionTreeNodePtr leaf) {
        leaf->scale = eng->GetAmplitude(i) / scale;
    });
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
            scale1 *= leaf1->scale;
            if (IS_NORM_0(scale1)) {
                break;
            }
        }
        for (j = 0; j < qubitCount; j++) {
            leaf2 = leaf2->branches[(i >> j) & 1U];
            scale2 *= leaf2->scale;
            if (IS_NORM_0(scale2)) {
                break;
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

    ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->Compose(toCopy, start); });

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}
void QBinaryDecisionTree::DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    Finish();

    if (dest) {
        dest->Dump();
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->Decompose(start, dest); });
    } else {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->Dispose(start, length); });
    }

    SetQubitCount(qubitCount - length);
}

real1_f QBinaryDecisionTree::Prob(bitLenInt qubitIndex)
{
    Finish();

    bitCapInt qPower = pow2(qubitIndex);

    int numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    par_for_skip(0, maxQPower, qPower, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt i = lcv | qPower;

        QBinaryDecisionTreeNodePtr leaf = root;
        complex scale = root->scale;
        for (bitLenInt j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
            scale *= leaf->scale;
            if (IS_NORM_0(scale)) {
                break;
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
    complex nrm = GetNonunitaryPhase();

    bitLenInt j;
    complex Y0;
    int bit;
    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < qPower; i++) {
        leaf = root;
        for (j = 0; j < qubit; j++) {
            bit = (i >> j) & 1U;
            leaf = leaf->branches[bit];
            if (IS_NORM_0(leaf->scale)) {
                break;
            }
        }

        if (IS_NORM_0(leaf->scale)) {
            continue;
        }

        if (result) {
            leaf->branches[0] = std::make_shared<QBinaryDecisionTreeNode>(ZERO_CMPLX);
            leaf->branches[1]->scale = nrm;
        } else {
            leaf->branches[0]->scale = nrm;
            leaf->branches[1] = std::make_shared<QBinaryDecisionTreeNode>(ZERO_CMPLX);
        }
    }

    root->Prune(qubitCount);

    return result;
}

void QBinaryDecisionTree::Apply2x2OnLeaf(const complex* mtrx, QBinaryDecisionTreeNodePtr leaf)
{
    if (IS_NORM_0(leaf->branches[0]->scale)) {
        leaf->branches[0] = leaf->branches[1]->ShallowClone();
        leaf->branches[0]->scale = ZERO_CMPLX;
    }

    if (IS_NORM_0(leaf->branches[1]->scale)) {
        leaf->branches[1] = leaf->branches[0]->ShallowClone();
        leaf->branches[1]->scale = ZERO_CMPLX;
    }

    // Apply gate.
    complex Y0 = leaf->branches[0]->scale;
    leaf->branches[0]->scale = mtrx[0] * Y0 + mtrx[1] * leaf->branches[1]->scale;
    leaf->branches[1]->scale = mtrx[2] * Y0 + mtrx[3] * leaf->branches[1]->scale;
}

void QBinaryDecisionTree::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    bitCapInt targetPow = pow2(target);
    std::shared_ptr<complex[]> mtrx(new complex[4]);
    std::copy(lMtrx, lMtrx + 4, mtrx.get());

    Dispatch(targetPow, [this, mtrx, target, targetPow]() {
        par_for(0, targetPow, [&](const bitCapInt& i, const int& cpu) {
            int bit;
            QBinaryDecisionTreeNodePtr leaf = root;

            // Iterate to qubit depth.
            for (bitLenInt j = 0; j < target; j++) {
                bit = (i >> j) & 1U;
                leaf = leaf->branches[bit];
                if (IS_NORM_0(leaf->scale)) {
                    break;
                }
            }

            if (!IS_NORM_0(leaf->scale)) {
                Apply2x2OnLeaf(mtrx.get(), leaf);
            }
        });

        root->Prune(qubitCount);
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

    std::unique_ptr<bitLenInt[]> sortedControls(new bitLenInt[controlLen]);
    std::shared_ptr<bitCapInt[]> qPowersSorted(new bitCapInt[controlLen]);
    std::copy(controls, controls + controlLen, sortedControls.get());
    std::sort(sortedControls.get(), sortedControls.get() + controlLen);

    bitCapInt lowControlMask = 0;
    bitLenInt c;
    for (c = 0; (c < controlLen) && (sortedControls[c] < target); c++) {
        qPowersSorted[c] = pow2(sortedControls[c]);
        lowControlMask |= qPowersSorted[c];
    }
    // TODO: This is a horrible kludge.
    if (c < controlLen) {
        // At least one control bit index is higher than the target.
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->ApplyControlledSingleBit(controls, controlLen, target, lMtrx); });
        return;
    }

    bitCapInt targetPow = pow2(target);

    Dispatch((targetPow >> controlLen),
        [this, mtrx, target, controlLen, lowControlMask, targetPow, qPowersSorted]() {
            // If any controls aren't set, skip.
            par_for_mask(0, targetPow, qPowersSorted.get(), controlLen, [&](const bitCapInt& lcv, const int& cpu) {
                bitCapInt i = lcv | lowControlMask;

                int iBit;
                QBinaryDecisionTreeNodePtr iLeaf = root;

                // Iterate to target bit.
                for (bitLenInt k = 0; k < target; k++) {
                    iBit = (i >> k) & 1U;
                    iLeaf = iLeaf->branches[iBit];
                    if (IS_NORM_0(iLeaf->scale)) {
                        return;
                    }
                }

                // (We can't handle controls with higher index than target, yet.
                Apply2x2OnLeaf(mtrx.get(), iLeaf);
            });

            root->Prune(qubitCount);
        });
}

} // namespace Qrack
