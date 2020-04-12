//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

#include <algorithm>
#include <future>
#include <map>
#include <mutex>
#include <set>

#include "common/parallel_for.hpp"
#include "common/qrack_types.hpp"

namespace Qrack {

class StateVectorArray : public StateVector {
protected:
    complex* amplitudes;

    static real1 normHelper(const complex& c) { return norm(c); }

    complex* Alloc(bitCapInt elemCount)
    {
// elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
#if defined(__APPLE__)
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE,
            ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
        return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
        return (complex*)_aligned_malloc(
            ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount,
            QRACK_ALIGN_SIZE);
#else
        return (complex*)aligned_alloc(QRACK_ALIGN_SIZE,
            ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
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
    StateVectorArray(bitCapInt cap)
        : StateVector(cap)
    {
        amplitudes = Alloc(capacity);
    }

    virtual ~StateVectorArray() { Free(); }

    complex read(const bitCapInt& i) { return amplitudes[i]; };

    void write(const bitCapInt& i, const complex& c) { amplitudes[i] = c; };

    void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2)
    {
        amplitudes[i1] = c1;
        amplitudes[i2] = c2;
    };

    void clear() { std::fill(amplitudes, amplitudes + capacity, ZERO_CMPLX); }

    void copy_in(const complex* copyIn) { std::copy(copyIn, copyIn + capacity, amplitudes); }

    void copy_out(complex* copyOut) { std::copy(amplitudes, amplitudes + capacity, copyOut); }

    void copy(StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorArray>(toCopy)); }

    void copy(StateVectorArrayPtr toCopy) { std::copy(toCopy->amplitudes, toCopy->amplitudes + capacity, amplitudes); }

    void get_probs(real1* outArray) { std::transform(amplitudes, amplitudes + capacity, outArray, normHelper); }

    bool is_sparse() { return false; }

    /// Returns empty if iteration should be over full set, otherwise just the iterable elements:
    std::set<bitCapInt> iterable(
        const bitCapInt& setMask, const bitCapInt& filterMask = 0, const bitCapInt& filterValues = 0)
    {
        return std::set<bitCapInt>();
    }
};

class StateVectorSparse : public StateVector, public ParallelFor {
protected:
    std::map<bitCapInt, complex> amplitudes;
    std::mutex mtx;

public:
    StateVectorSparse(bitCapInt cap)
        : StateVector(cap)
        , amplitudes()
    {
    }

    complex read(const bitCapInt& i)
    {
        complex toRet;
        mtx.lock();
        std::map<bitCapInt, complex>::const_iterator it = amplitudes.find(i);
        if (it == amplitudes.end()) {
            mtx.unlock();
            toRet = ZERO_CMPLX;
        } else {
            toRet = it->second;
            mtx.unlock();
        }
        return toRet;
    }

    void write(const bitCapInt& i, const complex& c)
    {
        if (norm(c) < min_norm) {
            mtx.lock();
            amplitudes.erase(i);
            mtx.unlock();
        } else {
            mtx.lock();
            amplitudes[i] = c;
            mtx.unlock();
        }
    }

    void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2)
    {
        if ((norm(c1) > min_norm) || (norm(c2) > min_norm)) {
            write(i1, c1);
            write(i2, c2);
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
        for (bitCapInt i = 0; i < capacity; i++) {
            write(i, copyIn[i]);
        }
    }

    void copy_out(complex* copyOut)
    {
        for (bitCapInt i = 0; i < capacity; i++) {
            copyOut[i] = read(i);
        }
    }

    void copy(const StateVectorPtr toCopy) { copy(std::dynamic_pointer_cast<StateVectorSparse>(toCopy)); }

    void copy(StateVectorSparsePtr toCopy)
    {
        mtx.lock();
        amplitudes = toCopy->amplitudes;
        mtx.unlock();
    }

    void get_probs(real1* outArray)
    {
        for (bitCapInt i = 0; i < capacity; i++) {
            outArray[i] = norm(read(i));
        }
    }

    bool is_sparse() { return (amplitudes.size() < (capacity >> 1U)); }

    /// Returns empty if iteration should be over full set, otherwise just the iterable elements:
    std::set<bitCapInt> iterable(
        const bitCapInt& setMask, const bitCapInt& filterMask = 0, const bitCapInt& filterValues = 0)
    {
        if ((filterMask == 0) && (filterValues != 0)) {
            return {};
        }

        int32_t i, combineCount;

        bitCapInt unsetMask = ~setMask;

        int32_t threadCount = GetConcurrencyLevel();
        std::vector<std::set<bitCapInt>> toRet(threadCount);
        std::vector<std::set<bitCapInt>>::iterator toRetIt;

        mtx.lock();

        if ((filterMask == 0) && (filterValues == 0)) {
            par_for(0, amplitudes.size(), [&](const bitCapInt lcv, const int cpu) {
                std::map<bitCapInt, complex>::const_iterator it = amplitudes.begin();
                std::advance(it, lcv);
                toRet[cpu].insert(it->first & unsetMask);
            });
        } else {
            bitCapInt unfilterMask = ~filterMask;

            par_for(0, amplitudes.size(), [&](const bitCapInt lcv, const int cpu) {
                std::map<bitCapInt, complex>::const_iterator it = amplitudes.begin();
                std::advance(it, lcv);
                if ((it->first & filterMask) == filterValues) {
                    toRet[cpu].insert(it->first & unsetMask & unfilterMask);
                }
            });
        }

        for (i = (toRet.size() - 1); i >= 0; i--) {
            if (toRet[i].size() == 0) {
                toRetIt = toRet.begin();
                std::advance(toRetIt, i);
                toRet.erase(toRetIt);
            }
        }

        if (toRet.size() == 0) {
            mtx.unlock();
            return {};
        }

        while (toRet.size() > 1U) {
            // Work odd unit into collapse sequence:
            if (toRet.size() & 1U) {
                toRet[toRet.size() - 2U].insert(toRet[toRet.size() - 1U].begin(), toRet[toRet.size() - 1U].end());
                toRet.pop_back();
            }

            combineCount = toRet.size() / 2U;
            std::vector<std::future<void>> futures(combineCount);
            for (i = (combineCount - 1U); i >= 0; i--) {
                futures[i] = std::async(std::launch::async, [i, &toRet]() {
                    toRet[i * 2U].insert(toRet[i * 2U + 1U].begin(), toRet[i * 2U + 1U].end());
                    toRet[i * 2U + 1U].clear();
                });
            }

            for (i = (combineCount - 1U); i >= 0; i--) {
                futures[i].get();
                toRetIt = toRet.begin();
                std::advance(toRetIt, i * 2U + 1U);
                toRet.erase(toRetIt);
            }
        }

        mtx.unlock();

        return toRet[0];
    }
};

} // namespace Qrack
