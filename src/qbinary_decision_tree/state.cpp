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

void QBinaryDecisionTree::SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG)
{
    qubitMap = QBdtQubitMap(qubitCount);

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phaseFac = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phaseFac = complex(ONE_R1, ZERO_R1);
        }
    }

    root = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
    QBinaryDecisionTreeNodePtr leaf = root;
    for (bitLenInt qubit = 0; qubit < qubitCount; qubit++) {
        leaf->Branch(1U, ZERO_CMPLX);
        leaf = leaf->branches[(initState >> qubit) & 1U];
        leaf->scale = phaseFac;
    }
}

QInterfacePtr QBinaryDecisionTree::Clone()
{
    QBinaryDecisionTree copyPtr = std::make_shared<QBinaryDecisionTree>(qubitCount, 0, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, amplitudeFloor);

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
            copyLeaf = std::make_shared<QBinaryDecisionTreeNode>(leaf.scale);
        }
    }
}

void QBinaryDecisionTree::GetTraversal(Fn getLambda)
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

    return toRet;
}
void QBinaryDecisionTree::SetTraversal(Fn setLambda)
{
    root->Branch(qubitCount);

    bitCapInt maxQPower = pow2(qubitCount);
    bitLenInt j;

    QBinaryDecisionTreeNodePtr leaf;
    for (bitCapInt i = 0; i < maxQPower; i++) {
        leaf = root;
        for (j = 0; j < qubitCount; j++) {
            leaf = leaf->branches[(i >> j) & 1U];
        }
        setLamda(i, leaf);
    }

    root->Prune();
}
StateVectorPtr QBinaryDecisionTree::ToStateVector(bool isSparse)
{
    // TODO: We can make sparse state vector array initialization more efficient.
    bitCapInt maxQPower = pow2(qubitCount);
    StateVectorPtr toRet = isSparse ? (StateVectorPtr)std::make_shared<StateVectorSparse>(maxQPower)
                                    : (StateVectorPtr)std::make_shared<StateVectorArray>(maxQPower);

    GetTraversal([toRet](bitCapInt i, complex scale) { toRet->write(qubitMap.mapPermutation(i), scale); });

    return toRet;
}
void QBinaryDecisionTree::FromStateVector(StateVectorPtr stateVec)
{
    qubitMap = QBdtQubitMap(qubitCount);
    root = std::make_shared<QBinaryDecisionTreeNode>();
    SetTraversal([stateVec](bitCapInt i, QBinaryDecisionTreeNode leaf) { leaf->scale = stateVec->read(i); });
}
void QBinaryDecisionTree::GetQuantumState(complex* state)
{
    GetTraversal([state](bitCapInt i, complex scale) { state[qubitMap.mapPermutation(i)] = scale; });
}
void QBinaryDecisionTree::SetQuantumState(const complex* state)
{
    qubitMap = QBdtQubitMap(qubitCount);
    root = std::make_shared<QBinaryDecisionTreeNode>();
    SetTraversal([state](bitCapInt i, QBinaryDecisionTreeNode leaf) { leaf->scale = state[i]; });
}
void QBinaryDecisionTree::GetProbs(real1* outputProbs)
{
    GetTraversal(
        [outputProbs](bitCapInt i, complex scale) { outputProbs[qubitMap.mapPermutation(i)] = scale * scale; });
}

complex QBinaryDecisionTree::GetAmplitude(bitCapInt perm)
{
    perm = qubitMap.mapPermutation(perm);
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
    perm = qubitMap.mapPermutation(perm);
    int bit;
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

    child.scale = amp / scale;

    if (IS_NORM_0(child.scale - leaf->branches[bit ^ 1U].scale)) {
        root->Prune(perm);
    }
}

bitLenInt QBinaryDecisionTree::Compose(QBinaryDecisionTree toCopy, bitLenInt start)
{
    if (start == 0) {
        QBinaryDecisionTreePtr clone = toCopy->Clone();
        std::swap(root, clone->root);
        clone->SetQubitCount(qubitCount);
        SetQubitCount(toCopy->qubitCount);
        SetTraversal([clone](bitCapInt i, QBinaryDecisionTreeNode leaf) {
            QBinaryDecisionTreePtr toCopyClone = clone->Clone();
            leaf->branches[0] = toCopyClone->root->branches[0];
            leaf->branches[1] = toCopyClone->root->branches[1];
        });
    } else if (start == qubitCount) {
        SetTraversal([toCopy](bitCapInt i, QBinaryDecisionTreeNode leaf) {
            QBinaryDecisionTreePtr toCopyClone = toCopy->Clone();
            leaf->branches[0] = toCopyClone->root->branches[0];
            leaf->branches[1] = toCopyClone->root->branches[1];
        });
    } else {
        throw std::runtime_error("Mid-range QBinaryDecisionTree::Compose() not yet implemented.");
    }

    SetQubitCount(qubitCount + toCopy->qubitCount);
}
void DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest)
{
    bitLenInt i;
    QBinaryDecisionTreeNodePtr leaf;
    QBinaryDecisionTreeNodePtr child = root;
    for (i = 0; i < start; i++) {
        leaf = child;
        child = leaf.branches[0];
        if (!child) {
            // All amplitudes must be the same.
            if (dest) {
                dest->root = randGlobalPhase ? std::polar(ONE_R1, 2 * PI_R1 * Rand()) : ONE_CMPLX;
            }
            break;
        }
    }

    // ANY child tree from this point is assumed to be EXACTLY EQUIVALENT for the length to Decompose().
    // WARNING: "Compose()" doesn't seem to need a normalization pass, but does this?

    if (child && dest) {
        dest->root = child;
        dest->SetQubitCount(qubitCount - start);
        QBinaryDecisionTreePtr clone = dest->Clone();
        dest->root = clone->root;

        if ((start + length) < qubitCount) {
            dest->Dispose(start + length, remainder);
        }
    }

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
        remainderLeaf = leaf for (j = 0; j < length; j++)
        {
            remainderLeaf = remainderLeaf->branches[(i >> j) & 1U];
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

} // namespace Qrack
