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
        if (depth == 0) {
            return;
        }

        if (!branches[0]) {
            branches[0] = std::make_shared<QBinaryDecisionTreeNode>(val);
        }
        if (!branches[1]) {
            branches[1] = std::make_shared<QBinaryDecisionTreeNode>(val);
        }

        if (depth) {
            branches[0]->Branch(depth - 1U, val);
            branches[1]->Branch(depth - 1U, val);
        }
    }

    void Prune(bitLenInt depth = bitsInCap)
    {
        depth--;
        if (branches[0]) {
            branches[0]->Prune(depth);
        }
        if (branches[1]) {
            branches[1]->Prune(depth);
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

    void Prune(bitCapInt perm, bitLenInt depth = bitsInCap)
    {
        if (!depth) {
            return;
        }

        bitCapInt bit = perm & 1U;
        perm >>= 1U;
        depth--;

        if (branches[bit]) {
            branches[bit]->Prune(perm, depth);
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

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest);

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

    virtual void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG);

    virtual QInterfacePtr Clone();

    virtual void GetQuantumState(complex* state);
    virtual void SetQuantumState(const complex* state);
    virtual void GetProbs(real1* outputProbs);

    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);

    virtual bitLenInt Compose(QBinaryDecisionTreePtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBinaryDecisionTree>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QBinaryDecisionTree>(dest));
    }
    virtual void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, NULL); }

    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        DecomposeDispose(start, length, NULL);
    }

    virtual real1_f Prob(bitLenInt qubitIndex);

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
};
} // namespace Qrack
