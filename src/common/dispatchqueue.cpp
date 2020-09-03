//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// From https://github.com/embeddedartistry/embedded-resources/blob/master/examples/cpp/dispatch.cpp

#include <chrono>

#include "config.h"
#include "dispatchqueue.hpp"

namespace Qrack {

DispatchQueue::DispatchQueue()
    : quit_(false)
    , isFinished_(true)
    , isStarted_(false)
{
}

DispatchQueue::~DispatchQueue()
{
    if (!isStarted_) {
        return;
    }

    std::unique_lock<std::mutex> lock(lock_);
    quit_ = true;
    lock.unlock();
    cv_.notify_all();

    // Wait for thread to finish before we exit
    thread_.get();

    isFinished_ = true;
    cvFinished_.notify_all();
}

void DispatchQueue::finish()
{
    if (!isStarted_) {
        return;
    }

    std::unique_lock<std::mutex> lock(lock_);
    cvFinished_.wait(lock, [this] { return isFinished_; });
}

void DispatchQueue::dump()
{
    if (!isStarted_) {
        return;
    }

    std::unique_lock<std::mutex> lock(lock_);
    std::queue<fp_t> empty;
    std::swap(q_, empty);
    isFinished_ = true;
    lock.unlock();
    cvFinished_.notify_all();
}

void DispatchQueue::dispatch(const fp_t& op)
{
    std::unique_lock<std::mutex> lock(lock_);
    q_.push(op);
    isFinished_ = false;
    if (!isStarted_) {
        isStarted_ = true;
        thread_ = std::async(std::launch::async, [this] { dispatch_thread_handler(); });
    }

    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lock.unlock();
    cv_.notify_one();
}

void DispatchQueue::dispatch(fp_t&& op)
{
    std::unique_lock<std::mutex> lock(lock_);
    q_.push(std::move(op));
    isFinished_ = false;
    if (!isStarted_) {
        isStarted_ = true;
        thread_ = std::async(std::launch::async, [this] { dispatch_thread_handler(); });
    }

    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lock.unlock();
    cv_.notify_one();
}

void DispatchQueue::dispatch_thread_handler(void)
{
    std::unique_lock<std::mutex> lock(lock_);

    do {
        // Wait until we have data or a quit signal
        cv_.wait(lock, [this] { return (q_.size() || quit_); });

        // after wait, we own the lock
        if (!quit_ && q_.size()) {
            auto op = std::move(q_.front());
            q_.pop();

            // unlock now that we're done messing with the queue
            lock.unlock();

            op();

            lock.lock();

            if (!q_.size()) {
                isFinished_ = true;
                cvFinished_.notify_all();
            }
        }
    } while (!quit_);
}

} // namespace Qrack
