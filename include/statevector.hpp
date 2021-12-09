//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This header defines buffers for Qrack::QFusion.
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/parallel_for.hpp"
#include "common/qrack_types.hpp"

#include <algorithm>
#include <mutex>
#include <set>

#if ENABLE_PTHREAD
#include <future>
#endif

#if ENABLE_UINT128
#if BOOST_AVAILABLE
#include <boost/functional/hash.hpp>
#include <unordered_map>
#define SparseStateVecMap std::unordered_map<bitCapIntOcl, complex>
#else
#include <map>
#define SparseStateVecMap std::map<bitCapIntOcl, complex>
#endif
#else
#if QBCAPPOW > 7
#include <boost/functional/hash.hpp>
#endif
#include <unordered_map>
#define SparseStateVecMap std::unordered_map<bitCapIntOcl, complex>
#endif

namespace Qrack {

class StateVectorArray : public StateVector {
public:
    complex* amplitudes;

protected:
    static real1_f normHelper(const complex& c) { return norm(c); }

    complex* Alloc(bitCapIntOcl elemCount)
    {
        size_t allocSize = sizeof(complex) * elemCount;
        if (allocSize < QRACK_ALIGN_SIZE) {
            allocSize = QRACK_ALIGN_SIZE;
        }
// elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
#if defined(__APPLE__)
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
        return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
        return (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE);
#elif defined(__ANDROID__)
        return (complex*)malloc(allocSize);
#else
        return (complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize);
#endif
    }

    virtual void Free()
    {
        if (amplitudes) {
#if defined(_WIN32)
            _aligned_free(amplitudes);
#else
            free(amplitudes);
#endif
        }
        amplitudes = NULL;
    }

public:
    StateVectorArray(bitCapIntOcl cap)
        : StateVector(cap)
    {
        amplitudes = Alloc(capacity);
    }

    virtual ~StateVectorArray() { Free(); }

    complex read(const bitCapIntOcl& i) { return amplitudes[i]; };

    void write(const bitCapIntOcl& i, const complex& c) { amplitudes[i] = c; };

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        amplitudes[i1] = c1;
        amplitudes[i2] = c2;
    };

    void clear() { std::fill(amplitudes, amplitudes + (bitCapIntOcl)capacity, ZERO_CMPLX); }

    void copy_in(const complex* copyIn)
    {
        if (copyIn) {
            std::copy(copyIn, copyIn + (bitCapIntOcl)capacity, amplitudes);
        } else {
            std::fill(amplitudes, amplitudes + (bitCapIntOcl)capacity, ZERO_CMPLX);
        }
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (copyIn) {
            std::copy(copyIn, copyIn + length, amplitudes + offset);
        } else {
            std::fill(amplitudes, amplitudes + length, ZERO_CMPLX);
        }
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        if (copyInSv) {
            const complex* copyIn = std::dynamic_pointer_cast<StateVectorArray>(copyInSv)->amplitudes + srcOffset;
            std::copy(copyIn, copyIn + length, amplitudes + dstOffset);
        } else {
            std::fill(amplitudes + dstOffset, amplitudes + dstOffset + length, ZERO_CMPLX);
        }
    }

    void copy_out(complex* copyOut) { std::copy(amplitudes, amplitudes + capacity, copyOut); }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        std::copy(amplitudes + offset, amplitudes + offset + capacity, copyOut);
    }

    void copy(StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorArray>(toCopy)); }

    void copy(StateVectorArrayPtr toCopy) { std::copy(toCopy->amplitudes, toCopy->amplitudes + capacity, amplitudes); }

    void shuffle(StateVectorPtr svp) { shuffle(std::dynamic_pointer_cast<StateVectorArray>(svp)); }

    void shuffle(StateVectorArrayPtr svp)
    {
        std::swap_ranges(amplitudes + (capacity >> ONE_BCI), amplitudes + capacity, svp->amplitudes);
    }

    void get_probs(real1* outArray) { std::transform(amplitudes, amplitudes + capacity, outArray, normHelper); }

    bool is_sparse() { return false; }
};

class StateVectorSparse : public StateVector, public ParallelFor {
protected:
    SparseStateVecMap amplitudes;
    std::mutex mtx;

    complex readUnlocked(const bitCapIntOcl& i)
    {
        auto it = amplitudes.find(i);
        return (it == amplitudes.end()) ? ZERO_CMPLX : it->second;
    }

    complex readLocked(const bitCapIntOcl& i)
    {
        mtx.lock();
        auto it = amplitudes.find(i);
        bool isFound = (it != amplitudes.end());
        mtx.unlock();
        return isFound ? it->second : ZERO_CMPLX;
    }

public:
    StateVectorSparse(bitCapIntOcl cap)
        : StateVector(cap)
        , amplitudes()
    {
    }

    complex read(const bitCapIntOcl& i) { return isReadLocked ? readLocked(i) : readUnlocked(i); }

    void write(const bitCapIntOcl& i, const complex& c)
    {
        bool isCSet = (c != ZERO_CMPLX);

        mtx.lock();

        auto it = amplitudes.find(i);
        bool isFound = (it != amplitudes.end());
        if (isCSet == isFound) {
            mtx.unlock();
            if (isCSet) {
                it->second = c;
            }
        } else {
            if (isCSet) {
                amplitudes[i] = c;
            } else {
                amplitudes.erase(it);
            }
            mtx.unlock();
        }
    }

    void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2)
    {
        bool isC1Set = (c1 != ZERO_CMPLX);
        bool isC2Set = (c2 != ZERO_CMPLX);
        if (!(isC1Set || isC2Set)) {
            return;
        }

        if (isC1Set && isC2Set) {
            mtx.lock();
            amplitudes[i1] = c1;
            amplitudes[i2] = c2;
            mtx.unlock();
        } else if (isC1Set) {
            mtx.lock();
            amplitudes.erase(i2);
            amplitudes[i1] = c1;
            mtx.unlock();
        } else {
            mtx.lock();
            amplitudes.erase(i1);
            amplitudes[i2] = c2;
            mtx.unlock();
        }
    }

    void clear()
    {
        mtx.lock();
        amplitudes.clear();
        mtx.unlock();
    }

    void copy_in(const complex* copyIn)
    {
        if (!copyIn) {
            clear();
            return;
        }

        mtx.lock();
        for (bitCapIntOcl i = 0; i < capacity; i++) {
            if (copyIn[i] == ZERO_CMPLX) {
                amplitudes.erase(i);
            } else {
                amplitudes[i] = copyIn[i];
            }
        }
        mtx.unlock();
    }

    void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        if (!copyIn) {
            mtx.lock();
            for (bitCapIntOcl i = 0; i < length; i++) {
                amplitudes.erase(i);
            }
            mtx.unlock();
            return;
        }

        mtx.lock();
        for (bitCapIntOcl i = 0; i < length; i++) {
            if (copyIn[i] == ZERO_CMPLX) {
                amplitudes.erase(i);
            } else {
                amplitudes[i + offset] = copyIn[i];
            }
        }
        mtx.unlock();
    }

    void copy_in(
        StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset, const bitCapIntOcl length)
    {
        StateVectorSparsePtr copyIn = std::dynamic_pointer_cast<StateVectorSparse>(copyInSv);

        if (!copyIn) {
            mtx.lock();
            for (bitCapIntOcl i = 0; i < length; i++) {
                amplitudes.erase(i + srcOffset);
            }
            mtx.unlock();
            return;
        }

        mtx.lock();
        for (bitCapIntOcl i = 0; i < length; i++) {
            complex amp = copyIn->read(i + srcOffset);
            if (amp == ZERO_CMPLX) {
                amplitudes.erase(i + srcOffset);
            } else {
                amplitudes[i + dstOffset] = amp;
            }
        }
        mtx.unlock();
    }

    void copy_out(complex* copyOut)
    {
        for (bitCapIntOcl i = 0; i < capacity; i++) {
            copyOut[i] = read(i);
        }
    }

    void copy_out(complex* copyOut, const bitCapIntOcl offset, const bitCapIntOcl length)
    {
        for (bitCapIntOcl i = 0; i < length; i++) {
            copyOut[i] = read(i + offset);
        }
    }

    void copy(const StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorSparse>(toCopy)); }

    void copy(StateVectorSparsePtr toCopy)
    {
        mtx.lock();
        amplitudes = toCopy->amplitudes;
        mtx.unlock();
    }

    void shuffle(StateVectorPtr svp) { shuffle(std::dynamic_pointer_cast<StateVectorSparse>(svp)); }

    void shuffle(StateVectorSparsePtr svp)
    {
        size_t halfCap = (size_t)(capacity >> ONE_BCI);
        mtx.lock();
        for (bitCapIntOcl i = 0; i < halfCap; i++) {
            complex amp = svp->read(i);
            svp->write(i, read(i + halfCap));
            write(i + halfCap, amp);
        }
        mtx.unlock();
    }

    void get_probs(real1* outArray)
    {
        for (bitCapIntOcl i = 0; i < capacity; i++) {
            outArray[i] = norm(read(i));
        }
    }

    bool is_sparse() { return (amplitudes.size() < (size_t)(capacity >> ONE_BCI)); }

    std::vector<bitCapIntOcl> iterable()
    {
        int64_t threadCount = GetConcurrencyLevel();
        std::vector<std::vector<bitCapIntOcl>> toRet(threadCount);
        std::vector<std::vector<bitCapIntOcl>>::iterator toRetIt;

        mtx.lock();

        par_for(0, amplitudes.size(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            auto it = amplitudes.begin();
            std::advance(it, lcv);
            toRet[cpu].push_back(it->first);
        });

        mtx.unlock();

        for (int64_t i = (int64_t)(toRet.size() - 1); i >= 0; i--) {
            if (toRet[i].size() == 0) {
                toRetIt = toRet.begin();
                std::advance(toRetIt, i);
                toRet.erase(toRetIt);
            }
        }

        if (toRet.size() == 0) {
            return {};
        }

        while (toRet.size() > 1U) {
            // Work odd unit into collapse sequence:
            if (toRet.size() & 1U) {
                toRet[toRet.size() - 2U].insert(
                    toRet[toRet.size() - 2U].end(), toRet[toRet.size() - 1U].begin(), toRet[toRet.size() - 1U].end());
                toRet.pop_back();
            }

            int64_t combineCount = (int64_t)toRet.size() / 2U;
#if ENABLE_PTHREAD
            std::vector<std::future<void>> futures(combineCount);
            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                futures[i] = std::async(std::launch::async, [i, combineCount, &toRet]() {
                    toRet[i].insert(toRet[i].end(), toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                    toRet[i + combineCount].clear();
                });
            }
            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                futures[i].get();
                toRet.pop_back();
            }
#else
            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                toRet[i].insert(toRet[i].end(), toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                toRet.pop_back();
            }
#endif
        }

        return toRet[0];
    }

    /// Returns empty if iteration should be over full set, otherwise just the iterable elements:
    std::set<bitCapIntOcl> iterable(
        const bitCapIntOcl& setMask, const bitCapIntOcl& filterMask = 0, const bitCapIntOcl& filterValues = 0)
    {
        if ((filterMask == 0) && (filterValues != 0)) {
            return {};
        }

        bitCapIntOcl unsetMask = ~setMask;

        int32_t threadCount = GetConcurrencyLevel();
        std::vector<std::set<bitCapIntOcl>> toRet(threadCount);
        std::vector<std::set<bitCapIntOcl>>::iterator toRetIt;

        mtx.lock();

        if ((filterMask == 0) && (filterValues == 0)) {
            par_for(0, amplitudes.size(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                auto it = amplitudes.begin();
                std::advance(it, lcv);
                toRet[cpu].insert(it->first & unsetMask);
            });
        } else {
            bitCapIntOcl unfilterMask = ~filterMask;

            par_for(0, amplitudes.size(), [&](const bitCapIntOcl lcv, const unsigned& cpu) {
                auto it = amplitudes.begin();
                std::advance(it, lcv);
                if ((it->first & filterMask) == filterValues) {
                    toRet[cpu].insert(it->first & unsetMask & unfilterMask);
                }
            });
        }

        mtx.unlock();

        for (int64_t i = (int64_t)(toRet.size() - 1); i >= 0; i--) {
            if (toRet[i].size() == 0) {
                toRetIt = toRet.begin();
                std::advance(toRetIt, i);
                toRet.erase(toRetIt);
            }
        }

        if (toRet.size() == 0) {
            return {};
        }

        while (toRet.size() > 1U) {
            // Work odd unit into collapse sequence:
            if (toRet.size() & 1U) {
                toRet[toRet.size() - 2U].insert(toRet[toRet.size() - 1U].begin(), toRet[toRet.size() - 1U].end());
                toRet.pop_back();
            }

            int64_t combineCount = (int32_t)(toRet.size()) / 2U;
#if ENABLE_PTHREAD
            std::vector<std::future<void>> futures(combineCount);
            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                futures[i] = std::async(std::launch::async, [i, combineCount, &toRet]() {
                    toRet[i].insert(toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                    toRet[i + combineCount].clear();
                });
            }

            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                futures[i].get();
                toRet.pop_back();
            }
#else
            for (int64_t i = (combineCount - 1U); i >= 0; i--) {
                toRet[i].insert(toRet[i + combineCount].begin(), toRet[i + combineCount].end());
                toRet.pop_back();
            }
#endif
        }

        return toRet[0];
    }
};

} // namespace Qrack
