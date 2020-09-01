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

#include "common/dispatchqueue.hpp"

namespace Qrack {

DispatchQueue::DispatchQueue(size_t thread_cnt)
    : threads_(thread_cnt)
{
    start();
}

DispatchQueue::~DispatchQueue() { dump(); }

void DispatchQueue::start()
{
    for (size_t i = 0; i < threads_.size(); i++) {
        threads_[i] = std::thread(&DispatchQueue::dispatch_thread_handler, this);
    }
}

void DispatchQueue::finish()
{
    // Signal to dispatch threads that it's time to wrap up
    std::unique_lock<std::mutex> lock(lock_);
    quit_ = true;
    lock.unlock();
    cv_.notify_all();

    // Wait for threads to finish before we exit
    for (size_t i = 0; i < threads_.size(); i++) {
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
}

void DispatchQueue::dump()
{
    std::unique_lock<std::mutex> lock(lock_);
    std::queue<fp_t> empty;
    std::swap(q_, empty);
    lock.unlock();
    finish();
}

void DispatchQueue::restart()
{
    finish();
    quit_ = false;
    start();
}

void DispatchQueue::dispatch(const fp_t& op)
{
    std::unique_lock<std::mutex> lock(lock_);
    q_.push(op);

    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lock.unlock();
    cv_.notify_one();
}

void DispatchQueue::dispatch(fp_t&& op)
{
    std::unique_lock<std::mutex> lock(lock_);
    q_.push(std::move(op));

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
        }
    } while (!quit_);

    while (q_.size()) {
        auto op = std::move(q_.front());
        q_.pop();

        // unlock now that we're done messing with the queue
        lock.unlock();

        op();

        lock.lock();
    }
}

} // namespace Qrack
