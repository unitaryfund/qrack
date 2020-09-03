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

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Qrack {

class DispatchQueue {
    typedef std::function<void(void)> fp_t;

public:
    DispatchQueue();
    ~DispatchQueue();

    // dispatch and copy
    void dispatch(const fp_t& op);
    // dispatch and move
    void dispatch(fp_t&& op);
    // finish queue
    void finish();
    // dump queue
    void dump();
    // check if queue is finished
    bool isFinished() { return isFinished_; }

    // Deleted operations
    DispatchQueue(const DispatchQueue& rhs) = delete;
    DispatchQueue& operator=(const DispatchQueue& rhs) = delete;
    DispatchQueue(DispatchQueue&& rhs) = delete;
    DispatchQueue& operator=(DispatchQueue&& rhs) = delete;

private:
    std::mutex lock_;
    std::thread thread_;
    std::queue<fp_t> q_;
    std::condition_variable cv_;
    std::condition_variable cvFinished_;
    bool quit_;
    bool isFinished_;

    void dispatch_thread_handler(void);
};

} // namespace Qrack
