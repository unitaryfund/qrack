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

#pragma once

#include "qinterface.hpp"
#include "statevector.hpp"

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

struct QBinaryDecisionTreeNode;
typedef std::shared_ptr<QBinaryDecisionTreeNode> QBinaryDecisionTreeNodePtr;

class QBinaryDecisionTree;
typedef std::shared_ptr<QBinaryDecisionTree> QBinaryDecisionTreePtr;

struct QBinaryDecisionTreeNode {
    complex scale;
    QBinaryDecisionTreeNodePtr branches[2];

    QBinaryDecisionTreeNode()
        : scale(ONE_CMPLX)
        , branches({ NULL, NULL })
    {
    }

    QBinaryDecisionTreeNode(complex val)
        : scale(val)
        , branches({ NULL, NULL })
    {
    }

    void Branch(bitLenInt depth = 1U, complex val = ONE_CMPLX)
    {
        branches[0] = std::make_shared<QBinaryDecisionTreeNode>(val);
        branches[1] = std::make_shared<QBinaryDecisionTreeNode>(val);

        if (depth) {
            branches[0]->Branch(depth - 1U, val);
            branches[1]->Branch(depth - 1U, val);
        }
    }

    void Prune()
    {
        if (branches[0]) {
            branches[0]->Prune();
        }
        if (branches[1]) {
            branches[1]->Prune();
        }

        if (!branches[0] || !branches[1]) {
            return;
        }

        if (branches[0]->branches[0] || branches[0]->branches[1] || branches[1]->branches[0] ||
            branches[1]->branches[1]) {
            return;
        }

        // We have 2 branches (with no children).
        if (IS_NORM_0(branches[0]->scale - branches[1]->scale)) {
            scale *= branches[0]->scale;
            branches[0] = NULL;
            branches[1] = NULL;
        }
    }

    void Prune(bitCapInt perm)
    {
        bitCapInt bit = perm & 1U;
        perm >>= 1U;
        if (branches[bit]) {
            branches[bit]->Prune(perm);
        }

        if (!branches[0] || !branches[1]) {
            return;
        }

        if (branches[0]->branches[0] || branches[0]->branches[1] || branches[1]->branches[0] ||
            branches[1]->branches[1]) {
            return;
        }

        // We have 2 branches (with no children).
        if (IS_NORM_0(branches[0]->scale - branches[1]->scale)) {
            scale *= branches[0]->scale;
            branches[0] = NULL;
            branches[1] = NULL;
        }
    }
};

class QBinaryDecisionTree : virtual public QInterface {
protected:
    QBinaryDecisionTreeNodePtr root;

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);

    StateVectorPtr ToStateVector(bool isSparse = false);
    void FromStateVector(StateVectorPtr stateVec);

public:
    QBinaryDecisionTree(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QBinaryDecisionTree(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QBinaryDecisionTree({}, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold, separation_thresh)
    {
    }

    void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG);

    void GetQuantumState(complex* state);
    void SetQuantumState(const complex* state);
    void GetProbs(real1* outputProbs);

    complex GetAmplitude(bitCapInt perm);
    void SetAmplitude(bitCapInt perm, complex amp);
};
} // namespace Qrack
