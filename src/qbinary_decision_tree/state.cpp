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

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

QBinaryDecisionTree::QBinaryDecisionTree(bitLenInt qbitCount, bitCapInt initState)
    : qubitCount(qbitCount)
    , root(NULL)
{
    SetPermutation(initState);
}

QBinaryDecisionTree::QBinaryDecisionTree(bitLenInt qbCount, StateVectorPtr stateVec)
    : qubitCount(qbCount)
    , root(NULL)
{
    FromStateVector(stateVec);
}

void QBinaryDecisionTree::SetPermutation(bitCapInt initState)
{
    root = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX);
    QBinaryDecisionTreeNodePtr leaf = root;
    for (bitLenInt qubit = 0; qubit < qubitCount; qubit++) {
        leaf->Branch(1U, ZERO_CMPLX);
        leaf = leaf->branches[(initState >> qubit) & 1U];
        leaf->scale = ONE_CMPLX;
    }
}

StateVectorPtr QBinaryDecisionTree::ToStateVector(bool isSparse)
{
    // TODO: We can make sparse state vector array initialization more efficient.
    bitCapInt maxQPower = pow2(qubitCount);
    StateVectorPtr toRet = isSparse ? (StateVectorPtr)std::make_shared<StateVectorSparse>(maxQPower)
                                    : (StateVectorPtr)std::make_shared<StateVectorArray>(maxQPower);
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
        toRet->write(i, scale);
    }

    return toRet;
}

void QBinaryDecisionTree::FromStateVector(StateVectorPtr stateVec)
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
        leaf->scale = stateVec->read(i);
    }

    root->Prune();
}
} // namespace Qrack
