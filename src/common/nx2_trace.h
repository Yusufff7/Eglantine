// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/common_types.h"

// Frame timeline for finding what a frame waits on (debugging). EDEN_NX2_TRACE=<first>:<last>
// records, from present <first> to present <last>, what the threads involved in a frame did:
// guest threads' supervisor calls (waits among them), GPU driver ioctls, the GPU threads' waits
// and work, and presents; then writes them as a Chrome trace (chrome://tracing, ui.perfetto.dev)
// to EDEN_NX2_TRACE_OUT (nx2_trace.json by default).
namespace Common::Nx2Trace {

struct Event {
    std::string name;
    const char* category;
    u64 begin_ns;
    u64 end_ns; ///< == begin_ns: an instant event
    u32 process; ///< 1 guest threads (thread = guest thread id), 2 host threads
    u64 thread;
    std::string args;
};

struct State {
    u64 first{1};
    u64 last{0};
    std::string out;
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    std::atomic<bool> active{};
    std::atomic<bool> written{};
    std::mutex mutex;
    std::vector<Event> events;
    std::map<std::pair<u32, u64>, std::string> thread_names;

    State() {
        const char* text = std::getenv("EDEN_NX2_TRACE");
        if (text == nullptr) {
            return;
        }
        const char* colon = std::strchr(text, ':');
        first = std::strtoull(text, nullptr, 10);
        last = colon != nullptr ? std::strtoull(colon + 1, nullptr, 10) : first + 120;
        const char* path = std::getenv("EDEN_NX2_TRACE_OUT");
        out = path != nullptr ? path : "nx2_trace.json";
        events.reserve(1 << 20);
    }
};

inline State& GetState() {
    static State state;
    return state;
}

/// Whether events are being recorded (cheap: check before building names).
inline bool Active() {
    return GetState().active.load(std::memory_order_relaxed);
}

inline u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - GetState().origin)
                                .count());
}

inline u64 HostThread() {
    return static_cast<u64>(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffffffff);
}

/// Names the calling host thread in the trace (call once per thread, any time).
inline void NameHostThread(std::string_view name) {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.thread_names[{2, HostThread()}] = std::string{name};
}

inline void NameGuestThread(u64 guest_thread, std::string_view name) {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.thread_names.try_emplace({1, guest_thread}, std::string{name});
}

inline void Record(Event&& event) {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.active.load(std::memory_order_relaxed)) {
        state.events.push_back(std::move(event));
    }
}

/// A span on the calling host thread.
inline void Span(const char* category, std::string name, u64 begin_ns, u64 end_ns,
                 std::string args = {}) {
    Record({std::move(name), category, begin_ns, end_ns, 2, HostThread(), std::move(args)});
}

/// A span on a guest thread.
inline void GuestSpan(const char* category, std::string name, u64 guest_thread, u64 begin_ns,
                      u64 end_ns, std::string args = {}) {
    Record({std::move(name), category, begin_ns, end_ns, 1, guest_thread, std::move(args)});
}

/// Measures a host-thread span from construction to destruction while recording.
class Scope {
public:
    Scope(const char* category_, const char* name_)
        : category{category_}, name{name_}, begin{Active() ? Now() : 0} {}
    ~Scope() {
        if (begin != 0 && Active()) {
            Span(category, name, begin, Now());
        }
    }

private:
    const char* category;
    const char* name;
    u64 begin;
};

inline void Write(State& state) {
    std::FILE* file = std::fopen(state.out.c_str(), "wb");
    if (file == nullptr) {
        return;
    }
    std::fputs("{\"traceEvents\":[\n", file);
    bool first_event = true;
    const auto separator = [&] {
        if (!first_event) {
            std::fputs(",\n", file);
        }
        first_event = false;
    };
    for (const auto& [key, name] : state.thread_names) {
        separator();
        std::fprintf(file,
                     "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%u,\"tid\":%llu,"
                     "\"args\":{\"name\":\"%s\"}}",
                     key.first, static_cast<unsigned long long>(key.second), name.c_str());
    }
    separator();
    std::fputs("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"guest\"}},"
               "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":2,\"args\":{\"name\":\"host\"}}",
               file);
    for (const auto& event : state.events) {
        separator();
        const double ts = static_cast<double>(event.begin_ns) / 1000.0;
        if (event.end_ns == event.begin_ns) {
            std::fprintf(file,
                         "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"i\",\"s\":\"g\",\"ts\":%.3f,"
                         "\"pid\":%u,\"tid\":%llu%s%s%s}",
                         event.name.c_str(), event.category, ts, event.process,
                         static_cast<unsigned long long>(event.thread),
                         event.args.empty() ? "" : ",\"args\":{\"info\":\"",
                         event.args.c_str(), event.args.empty() ? "" : "\"}");
        } else {
            std::fprintf(file,
                         "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,"
                         "\"pid\":%u,\"tid\":%llu%s%s%s}",
                         event.name.c_str(), event.category, ts,
                         static_cast<double>(event.end_ns - event.begin_ns) / 1000.0,
                         event.process, static_cast<unsigned long long>(event.thread),
                         event.args.empty() ? "" : ",\"args\":{\"info\":\"",
                         event.args.c_str(), event.args.empty() ? "" : "\"}");
        }
    }
    std::fputs("\n]}\n", file);
    std::fclose(file);
    std::fprintf(stderr, "nx2 trace: %zu events written to %s\n", state.events.size(),
                 state.out.c_str());
}

/// Guest presents so far (for logs that say when something happened).
inline std::atomic<u64> g_present_count{};

/// Called on each guest present with the count so far: starts and ends the recording.
inline void OnPresent(u64 count) {
    g_present_count.store(count, std::memory_order_relaxed);
    auto& state = GetState();
    if (state.first > state.last || state.written.load(std::memory_order_relaxed)) {
        return;
    }
    if (count == state.first) {
        state.active.store(true, std::memory_order_relaxed);
    }
    if (state.active.load(std::memory_order_relaxed)) {
        Record({"present " + std::to_string(count), "present", Now(), Now(), 2, HostThread(), {}});
    }
    if (count == state.last && state.active.exchange(false)) {
        state.written.store(true);
        std::scoped_lock lock{state.mutex};
        Write(state);
        state.events.clear();
        state.events.shrink_to_fit();
    }
}

} // namespace Common::Nx2Trace
