// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/thread.h"

namespace VideoCommon {

/// Destroys GPU objects on a thread of its own. Freeing large images and buffers (the driver
/// releasing hundreds of MiB) held the GPU thread for 20-100 ms at frame ends: whole frames lost.
/// Owners hand over objects the GPU no longer uses and Wait() before the device goes away.
class BackgroundDestroyer {
public:
    static BackgroundDestroyer& Instance() {
        static BackgroundDestroyer instance;
        return instance;
    }

    /// EDEN_NX2_BG_DESTROY=0: destroy on the calling thread, as before.
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_BG_DESTROY");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    template <typename T>
    void Destroy(std::vector<T>&& objects) {
        if (objects.empty()) {
            return;
        }
        if (!Enabled()) {
            objects.clear();
            return;
        }
        Push(std::make_unique<Batch<T>>(std::move(objects)));
    }

    /// Runs `func` on the destroyer thread (slow driver queries the GPU thread need not wait on).
    template <typename Func>
    void Run(Func&& func) {
        if (!Enabled()) {
            func();
            return;
        }
        Push(std::make_unique<Job<std::decay_t<Func>>>(std::forward<Func>(func)));
    }

    /// Returns once everything handed over so far is destroyed.
    void Wait() {
        std::unique_lock lock{mutex};
        idle.wait(lock, [this] { return queue.empty() && !busy; });
    }

private:
    struct BatchBase {
        virtual ~BatchBase() = default;
    };
    template <typename T>
    struct Batch final : BatchBase {
        explicit Batch(std::vector<T>&& objects_) : objects{std::move(objects_)} {}
        std::vector<T> objects;
    };

    template <typename Func>
    struct Job final : BatchBase {
        explicit Job(Func&& func_) : func{std::move(func_)} {}
        ~Job() override {
            func();
        }
        Func func;
    };

    BackgroundDestroyer() = default;

    void Push(std::unique_ptr<BatchBase> batch) {
        {
            std::scoped_lock lock{mutex};
            queue.push_back(std::move(batch));
            if (!thread.joinable()) {
                thread = std::jthread([this](std::stop_token token) { Loop(token); });
            }
        }
        work.notify_one();
    }

    ~BackgroundDestroyer() {
        if (thread.joinable()) {
            thread.request_stop();
            work.notify_one();
            thread.join();
        }
    }

    void Loop(std::stop_token token) {
        Common::SetCurrentThreadName("GPUDestroyer");
        std::unique_lock lock{mutex};
        while (true) {
            work.wait(lock, [&] { return !queue.empty() || token.stop_requested(); });
            if (queue.empty()) {
                return;
            }
            std::unique_ptr<BatchBase> batch = std::move(queue.front());
            queue.pop_front();
            busy = true;
            lock.unlock();
            batch.reset();
            lock.lock();
            busy = false;
            if (queue.empty()) {
                idle.notify_all();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable work;
    std::condition_variable idle;
    std::deque<std::unique_ptr<BatchBase>> queue;
    bool busy{};
    std::jthread thread;
};

} // namespace VideoCommon
