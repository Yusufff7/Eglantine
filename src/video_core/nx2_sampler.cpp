// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nx2_sampler.h"

#ifdef _WIN32
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <windows.h>
#include <dbghelp.h>
#include <timeapi.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/nx2_trace.h"
#include "common/thread.h"
#endif

namespace VideoCommon::Nx2Sampler {

#ifdef _WIN32
namespace {

constexpr size_t MaxFrames = 24;

struct Sample {
    std::array<u64, MaxFrames> frames;
    u32 count;
};

/// Walks the suspended thread's stack with the unwind tables only: nothing here may allocate or
/// take a lock the sampled thread could be holding.
u32 Walk(CONTEXT context, std::array<u64, MaxFrames>& frames) {
    u32 count = 0;
    while (count < MaxFrames && context.Rip != 0) {
        frames[count++] = context.Rip;
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (function == nullptr) {
            // Leaf function: the return address is at the top of the stack.
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += 8;
            continue;
        }
        PVOID handler_data = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                         &handler_data, &establisher, nullptr);
    }
    return count;
}

std::string Symbol(u64 address) {
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512]{};
    auto* const symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 511;
    DWORD64 displacement = 0;
    if (SymFromAddr(GetCurrentProcess(), address, &displacement, symbol)) {
        return std::string(symbol->Name, symbol->NameLen);
    }
    return fmt::format("{:#x}", address);
}

void Run(HANDLE thread, std::string name, u64 first, u64 last) {
    Common::SetCurrentThreadName("Nx2Sampler");
    std::vector<Sample> samples;
    samples.reserve(200'000);
    while (Common::Nx2Trace::g_present_count.load(std::memory_order_relaxed) < first) {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
    timeBeginPeriod(1);
    while (Common::Nx2Trace::g_present_count.load(std::memory_order_relaxed) < last &&
           samples.size() < samples.capacity()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
            break;
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        Sample sample{};
        if (GetThreadContext(thread, &context)) {
            sample.count = Walk(context, sample.frames);
        }
        ResumeThread(thread);
        if (sample.count != 0) {
            samples.push_back(sample);
        }
    }
    timeEndPeriod(1);

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    std::unordered_map<u64, std::string> names;
    const auto name_of = [&](u64 address) -> const std::string& {
        auto [it, is_new] = names.try_emplace(address);
        if (is_new) {
            it->second = Symbol(address);
        }
        return it->second;
    };
    std::unordered_map<std::string, u32> self;
    std::unordered_map<std::string, u32> total;
    std::unordered_map<std::string, u32> callers; ///< "callee <- caller" for the top self entries
    std::unordered_map<std::string, u32> chains;  ///< Five frames deep
    for (const Sample& sample : samples) {
        ++self[name_of(sample.frames[0])];
        std::unordered_set<std::string> seen;
        for (u32 i = 0; i < sample.count; ++i) {
            const std::string& function = name_of(sample.frames[i]);
            if (seen.insert(function).second) {
                ++total[function];
            }
        }
        if (sample.count > 1) {
            ++callers[name_of(sample.frames[0]) + "  <-  " + name_of(sample.frames[1])];
        }
        std::string chain = name_of(sample.frames[0]);
        for (u32 i = 1; i < (std::min)(sample.count, 5u); ++i) {
            chain += "  <-  " + name_of(sample.frames[i]);
        }
        ++chains[chain];
    }
    const auto top = [](const std::unordered_map<std::string, u32>& counts, size_t n) {
        std::vector<std::pair<u32, std::string>> sorted;
        for (const auto& [function, count] : counts) {
            sorted.emplace_back(count, function);
        }
        std::sort(sorted.rbegin(), sorted.rend());
        sorted.resize((std::min)(n, sorted.size()));
        return sorted;
    };
    const double scale = samples.empty() ? 0.0 : 100.0 / static_cast<double>(samples.size());
    LOG_INFO(HW_GPU, "Sampler: {} samples of {} (presents {}-{})", samples.size(), name, first, last);
    for (const auto& [count, function] : top(self, 60)) {
        LOG_INFO(HW_GPU, "  self {:5.1f}%  {}", count * scale, function);
    }
    for (const auto& [count, function] : top(total, 80)) {
        LOG_INFO(HW_GPU, "  total {:5.1f}%  {}", count * scale, function);
    }
    for (const auto& [count, pair] : top(callers, 60)) {
        LOG_INFO(HW_GPU, "  edge {:5.1f}%  {}", count * scale, pair);
    }
    for (const auto& [count, chain] : top(chains, 60)) {
        LOG_INFO(HW_GPU, "  chain {:5.1f}%  {}", count * scale, chain);
    }
    SymCleanup(GetCurrentProcess());
}

} // namespace
#endif

void Start(std::thread::native_handle_type thread, const char* name) {
#ifdef _WIN32
    const char* value = std::getenv("EDEN_NX2_SAMPLE");
    if (value == nullptr) {
        return;
    }
    // EDEN_NX2_SAMPLE_THREAD: which thread (back by default, or front).
    const char* which = std::getenv("EDEN_NX2_SAMPLE_THREAD");
    if (std::string{which != nullptr ? which : "back"} != name) {
        return;
    }
    char* end = nullptr;
    const u64 first = std::strtoull(value, &end, 10);
    const u64 last = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 10) : first + 600;
    HANDLE handle = nullptr;
    DuplicateHandle(GetCurrentProcess(), static_cast<HANDLE>(thread), GetCurrentProcess(), &handle,
                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
    std::thread([handle, name = std::string{name}, first, last] { Run(handle, name, first, last); })
        .detach();
#else
    (void)thread;
    (void)name;
#endif
}

void StartForCurrentThread(const char* name) {
#ifdef _WIN32
    Start(GetCurrentThread(), name);
#else
    (void)name;
#endif
}

} // namespace VideoCommon::Nx2Sampler
