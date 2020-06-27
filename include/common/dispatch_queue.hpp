// Adapted from https://embeddedartistry.com/blog/2017/02/08/implementing-an-asynchronous-dispatch-queue/
// and https://github.com/embeddedartistry/embedded-resources/blob/master/examples/cpp/dispatch.cpp

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace Qrack {

class DispatchQueue {
    typedef std::function<void(void)> fp_t;

public:
    DispatchQueue(std::string name, size_t thread_cnt = 1);
    ~DispatchQueue();

    // dispatch and copy
    void Dispatch(const fp_t& op);
    // dispatch and move
    void Dispatch(fp_t&& op);
    // wait for all items in dispatch
    void WaitAll();

    // Deleted operations
    DispatchQueue(const DispatchQueue& rhs) = delete;
    DispatchQueue& operator=(const DispatchQueue& rhs) = delete;
    DispatchQueue(DispatchQueue&& rhs) = delete;
    DispatchQueue& operator=(DispatchQueue&& rhs) = delete;

private:
    std::string name_;
    std::mutex lock_;
    std::vector<std::thread> threads_;
    std::queue<fp_t> q_;
    std::condition_variable cv_;
    bool quit_ = false;

    void dispatch_thread_handler(void);
};

} // namespace Qrack
