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
#include <map>
#include <mutex>

#include "common/qrack_types.hpp"

namespace Qrack {

class StateVector;
class StateVectorArray;
class StateVectorSparse;

typedef std::shared_ptr<StateVector> StateVectorPtr;
typedef std::shared_ptr<StateVectorArray> StateVectorArrayPtr;
typedef std::shared_ptr<StateVectorSparse> StateVectorSparsePtr;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
class StateVector {
protected:
    bitCapInt capacity;

public:
    StateVector(bitCapInt cap)
        : capacity(cap)
    {
    }
    virtual complex read(const bitCapInt& i) = 0;
    virtual void write(const bitCapInt& i, const complex& c) = 0;
    /// Optimized "write" that is only guaranteed to write if either amplitude is nonzero. (Useful for the result of 2x2
    /// tensor slicing.)
    virtual void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2) = 0;
    virtual void clear() = 0;
    virtual void copy_in(const complex* inArray) = 0;
    virtual void copy_out(complex* outArray) = 0;
    virtual void copy(const StateVector& toCopy) = 0;
    virtual void get_probs(real1* outArray) = 0;
};

class StateVectorArray : public StateVector {
protected:
    complex* amplitudes;

    static real1 normHelper(complex c) { return norm(c); }

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

    ~StateVectorArray() { Free(); }

    complex read(const bitCapInt& i) { return amplitudes[i]; };

    void write(const bitCapInt& i, const complex& c) { amplitudes[i] = c; };

    void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2)
    {
        amplitudes[i1] = c1;
        amplitudes[i2] = c2;
    };

    void clear() { std::fill(amplitudes, amplitudes + capacity, complex(ZERO_R1, ZERO_R1)); }

    void copy_in(const complex* copyIn) { std::copy(copyIn, copyIn + capacity, amplitudes); }

    void copy_out(complex* copyOut) { std::copy(amplitudes, amplitudes + capacity, copyOut); }

    void copy(const StateVector& toCopy) { copy((const StateVectorArray&)toCopy); }

    void copy(const StateVectorArray& toCopy)
    {
        if (capacity != toCopy.capacity) {
            Free();
            capacity = toCopy.capacity;
            Alloc(capacity);
        }
        std::copy(toCopy.amplitudes, toCopy.amplitudes + capacity, amplitudes);
    }

    void get_probs(real1* outArray) { std::transform(amplitudes, amplitudes + capacity, outArray, normHelper); }
};

class StateVectorSparse : public StateVector {
protected:
    std::map<bitCapInt, complex> amplitudes;
    std::recursive_mutex mtx;

public:
    StateVectorSparse(bitCapInt cap)
        : StateVector(cap)
        , amplitudes()
    {
    }

    complex read(const bitCapInt& i)
    {
        mtx.lock();
        complex toRet;
        std::map<bitCapInt, complex>::const_iterator it = amplitudes.find(i);
        if (it == amplitudes.end()) {
            toRet = complex(ZERO_R1, ZERO_R1);
        } else {
            toRet = it->second;
        }
        mtx.unlock();
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

    void copy(const StateVector& toCopy) { copy((const StateVectorSparse&)toCopy); }

    void copy(const StateVectorSparse& toCopy)
    {
        capacity = toCopy.capacity;
        mtx.lock();
        amplitudes = toCopy.amplitudes;
        mtx.unlock();
    }

    void get_probs(real1* outArray)
    {
        mtx.lock();
        for (bitCapInt i = 0; i < capacity; i++) {
            outArray[i] = norm(read(i));
        }
        mtx.unlock();
    }
};

} // namespace Qrack
