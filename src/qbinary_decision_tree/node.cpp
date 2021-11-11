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

#include "qbinary_decision_tree_node.hpp"

#include <future>
#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

void QBinaryDecisionTreeNode::Prune(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Prune recursively to depth.
    depth--;
    branches[0]->Prune(depth);
    if (b0 != b1) {
        branches[1]->Prune(depth);
    }

    complex phaseFac = std::polar(ONE_R1, (real1)(IS_NORM_0(b0->scale) ? std::arg(b1->scale) : std::arg(b0->scale)));
    scale *= phaseFac;
    b0->scale /= phaseFac;
    if (b0 == b1) {
        // Combining branches is the only other thing we try, below.
        return;
    }
    b1->scale /= phaseFac;

    // Now, we try to combine pointers to equivalent branches.

    bitCapInt depthPow = ONE_BCI << depth;

    // Combine single elements at bottom of full depth, up to where branches are equal below:
    par_for_qbdt(0, depthPow, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf0 = b0;
        QBinaryDecisionTreeNodePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        size_t bit = 0U;
        bitLenInt j;

        for (j = 0; j < depth; j++) {
            bit = SelectBit(i, depth - (j + 1U));

            if (!leaf0 || !leaf1 || (leaf0->branches[bit] == leaf1->branches[bit])) {
                break;
            }

            scale0 = leaf0->scale;
            leaf0 = leaf0->branches[bit];

            scale1 = leaf1->scale;
            leaf1 = leaf1->branches[bit];
        }

        if (!leaf0 || !leaf1 || (leaf0->branches[bit] != leaf1->branches[bit])) {
            return (bitCapIntOcl)0U;
        }

        if (IS_NORM_0(scale0 - scale1)) {
            leaf1->branches[bit] = leaf0->branches[bit];
        }

        // WARNING: Mutates loop control variable!
        return (bitCapIntOcl)((ONE_BCI << (bitCapIntOcl)(depth - j)) - ONE_BCI);
    });

    bool isSameAtTop = true;

    // Combine all elements at top of depth, as my 2 direct descendent branches:
    par_for_qbdt(0, depthPow, [&](const bitCapIntOcl& i, const unsigned& cpu) {
        QBinaryDecisionTreeNodePtr leaf0 = b0;
        QBinaryDecisionTreeNodePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        size_t bit = 0U;
        bitLenInt j;

        for (j = 0; j < depth; j++) {
            bit = SelectBit(i, depth - (j + 1U));

            if (leaf0) {
                scale0 *= leaf0->scale;
                leaf0 = leaf0->branches[bit];
            }

            if (leaf1) {
                scale1 *= leaf1->scale;
                leaf1 = leaf1->branches[bit];
            }

            if (leaf0 == leaf1) {
                break;
            }
        }

        if ((leaf0 != leaf1) || !IS_NORM_0(scale0 - scale1)) {
            // We can't combine our immediate children within depth.
            isSameAtTop = false;
            return depthPow;
        }

        // WARNING: Mutates loop control variable!
        return (bitCapIntOcl)((ONE_BCI << (depth - j)) - ONE_BCI);
    });

    // The branches terminate equal, within depth.
    if (isSameAtTop) {
        b1 = b0;
    }
}

void QBinaryDecisionTreeNode::Branch(bitLenInt depth, bool isZeroBranch)
{
    if (!depth) {
        return;
    }
    if (!isZeroBranch && IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    if (!b0) {
        b0 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
        b1 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
    } else {
        // Split all clones.
        b0 = b0->ShallowClone();
        b1 = b1->ShallowClone();
    }

    b0->Branch(depth - 1U, isZeroBranch);
    b1->Branch(depth - 1U, isZeroBranch);
}

void QBinaryDecisionTreeNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    b0->Normalize(depth - 1U);
    if (b0 != b1) {
        b1->Normalize(depth - 1U);
    }

    real1 nrm = (real1)sqrt(norm(b0->scale) + norm(b1->scale));
    b0->scale *= ONE_R1 / nrm;
    if (b0 != b1) {
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBinaryDecisionTreeNode::ConvertStateVector(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Depth-first
    depth--;
    b0->ConvertStateVector(depth);
    if (b0 != b1) {
        b1->ConvertStateVector(depth);
    }

    real1 nrm0 = norm(b0->scale);
    real1 nrm1 = norm(b1->scale);

    if ((nrm0 + nrm1) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (nrm0 <= FP_NORM_EPSILON) {
        scale = b1->scale;
        b0->SetZero();
        b1->scale = ONE_CMPLX;
        return;
    }

    if (nrm1 <= FP_NORM_EPSILON) {
        scale = b0->scale;
        b0->scale = ONE_CMPLX;
        b1->SetZero();
        return;
    }

    scale = std::polar((real1)sqrt(nrm0 + nrm1), (real1)std::arg(b0->scale));
    b0->scale /= scale;
    b1->scale /= scale;
}

// TODO: Find some way to abstract this with ParallelFor. (It's a duplicate method.)
// The reason for this design choice is that the memory per node for "Stride" and "numCores" attributes are on order of
// all other RAM per node in total. Remember that trees are recursively constructed with exponential scaling, and the
// memory per node should be thought of as akin to the memory per Schrödinger amplitude.
void QBinaryDecisionTreeNode::par_for_qbdt(const bitCapIntOcl begin, const bitCapIntOcl end, IncrementFunc fn)
{
    bitCapIntOcl itemCount = end - begin;

    const bitCapIntOcl Stride = (ONE_BCI << (bitCapIntOcl)10U);
    const unsigned numCores = std::thread::hardware_concurrency();

    if (itemCount < (Stride * numCores)) {
        bitCapIntOcl maxLcv = begin + itemCount;
        for (bitCapIntOcl j = begin; j < maxLcv; j++) {
            j |= fn(j, 0);
        }
        return;
    }

    bitCapIntOcl idx = 0;
    std::vector<std::future<void>> futures(numCores);
    std::mutex updateMutex;
    for (unsigned cpu = 0; cpu != numCores; ++cpu) {
        futures[cpu] = std::async(std::launch::async, [cpu, &idx, &begin, &itemCount, &updateMutex, fn]() {
            bitCapIntOcl i, j, l, maxJ;
            bitCapIntOcl k = 0;
            for (;;) {
                if (true) {
                    std::lock_guard<std::mutex> updateLock(updateMutex);
                    i = idx++;
                }
                l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (j = 0; j < maxJ; j++) {
                    k = j + l;
                    k |= fn(begin + k, cpu);
                    j = k - l;
                }
                i = k / Stride;
                if (i > idx) {
                    std::lock_guard<std::mutex> updateLock(updateMutex);
                    if (i > idx) {
                        idx = i;
                    }
                }
            }
        });
    }

    for (unsigned cpu = 0; cpu != numCores; ++cpu) {
        futures[cpu].get();
    }
}

} // namespace Qrack
