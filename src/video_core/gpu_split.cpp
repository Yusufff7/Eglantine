// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <string>
#include <cstdlib>
#include <thread>
#include <typeinfo>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstdio>

#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "common/nx2_trace.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_sampler.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};
std::atomic<bool> g_back_busy{};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    if (!g_back_busy.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::fprintf(stderr, "GPU split owner check: %s on thread %lu while the back runs\n", where,
                 thread);
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        std::fprintf(stderr, "  frame %s+%#llx\n", base_name ? base_name + 1 : name,
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                                                     reinterpret_cast<uintptr_t>(module)));
    }
    std::fflush(stderr);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    Nx2Sampler::Start(back_thread.native_handle(), "back");
    stall_thread = std::jthread([this](std::stop_token stop_token) { StallWatch(stop_token); });
    front_engine = nullptr;
    batch_engine = nullptr;
    batch_size = 0;
    run_engine = nullptr;
    run_size = 0;
    active = true;
#ifdef ARCHITECTURE_x86_64
    // Front macros run compiled (macro_turing_jit.h) only.
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
#endif
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    if (stall_thread.joinable()) {
        stall_thread.request_stop();
        stall_thread.join();
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::FlushState() {
    const size_t count = batch_size;
    batch_size = 0;
    // Named every time: cheap next to the batch, and the back never applies one to an engine
    // an earlier entry left selected.
    SwitchEngine(batch_engine);
    u64* const entry = Reserve(1 + count);
    entry[0] = static_cast<u64>(Kind::State) | (u64{count} << 8);
    std::memcpy(entry + 1, batch.data(), count * sizeof(u64));
    Commit(1 + count);
}

void GpuSplit::FlushRun() {
    const size_t count = run_size;
    run_size = 0;
    SwitchEngine(run_engine);
    u64* const entry = Reserve(1 + count);
    entry[0] = static_cast<u64>(Kind::Run) | (u64{count} << 8);
    std::memcpy(entry + 1, run.data(), count * sizeof(u64));
    Commit(1 + count);
}

void GpuSplit::Publish() {
    if (batch_size != 0) {
        FlushState();
    }
    if (run_size != 0) {
        FlushRun();
    }
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

namespace {

struct DrainStats {
    std::array<u64, static_cast<size_t>(GpuSplit::DrainReason::Count)> calls{};
    std::array<u64, static_cast<size_t>(GpuSplit::DrainReason::Count)> waits{};
    std::array<u64, static_cast<size_t>(GpuSplit::DrainReason::Count)> wait_ns{};
    std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

bool DrainStatsEnabled() {
    static const bool enabled = std::getenv("EDEN_NX2_DRAIN_STATS") != nullptr;
    return enabled;
}

// Front thread only: logs every 5 s how often each reason waited and for how long.
void RecordDrain(GpuSplit::DrainReason reason, bool waited, u64 ns) {
    static DrainStats stats;
    const auto index = static_cast<size_t>(reason);
    ++stats.calls[index];
    stats.waits[index] += waited ? 1 : 0;
    stats.wait_ns[index] += ns;
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.window_start < std::chrono::seconds{5}) {
        return;
    }
    static constexpr std::array names{"other", "acquire", "acquire full", "acquire queued", "acquire met", "acquire resume", "puller", "dma read",
                                      "dma read full", "sync memory", "front init"};
    std::string line;
    for (size_t i = 0; i < names.size(); ++i) {
        if (stats.calls[i] != 0) {
            line += fmt::format(" | {} {}/s waited {}/s {:.1f} ms/s", names[i], stats.calls[i] / 5,
                                stats.waits[i] / 5, stats.wait_ns[i] / 5e6);
        }
    }
    LOG_INFO(HW_GPU, "Drains per second:{}", line);
    stats = {};
    stats.window_start = now;
}

} // Anonymous namespace

void GpuSplit::Drain(DrainReason reason) {
    if (batch_size != 0) {
        FlushState();
    }
    if (run_size != 0) {
        FlushRun();
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    static constexpr std::array trace_names{
        "drain: other",       "drain: acquire",    "drain: acquire full", "drain: acquire queued",
        "drain: acquire met", "drain: acquire resume", "drain: puller",   "drain: dma read",
        "drain: dma read full", "drain: sync memory", "drain: front init"};
    Common::Nx2Trace::Scope trace{
        "split", consumed_cache == write_position
                     ? "drain (idle)"
                     : trace_names[std::min<size_t>(static_cast<size_t>(reason),
                                                    trace_names.size() - 1)]};
    if (consumed_cache == write_position) {
        if (DrainStatsEnabled()) {
            RecordDrain(reason, false, 0);
        }
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const auto start = DrainStatsEnabled() ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    const auto drain_start = std::chrono::steady_clock::now();
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = 8;
            if (DrainStatsEnabled()) {
                RecordDrain(reason, true,
                            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count()));
            }
            if (const auto took = std::chrono::steady_clock::now() - drain_start;
                took > std::chrono::milliseconds{30}) [[unlikely]] {
                static u32 reported{};
                if (reported++ < 200) {
                    LOG_WARNING(HW_GPU, "GPU front waited {} ms for the back (drain reason {})",
                                std::chrono::duration_cast<std::chrono::milliseconds>(took).count(),
                                static_cast<u32>(reason));
                }
            }
            return;
        }
        if (spins < 16384) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::DrainTo(u64 position, DrainReason reason) {
    if (position >= write_position) {
        return Drain(reason);
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= position) {
        if (DrainStatsEnabled()) {
            RecordDrain(reason, false, 0);
        }
        return;
    }
    if (published_local < position) {
        Publish();
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const auto start = DrainStatsEnabled() ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= position) {
            if (DrainStatsEnabled()) {
                RecordDrain(reason, true,
                            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count()));
            }
            return;
        }
        if (spins < 16384) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

} // namespace VideoCommon

bool Tegra::Nx2SemaphoreWaiters::TrackWriters() {
    return VideoCommon::GpuSplit::WaitStatsEnabled();
}

namespace VideoCommon {

bool GpuSplit::WaitStatsEnabled() {
    static const bool enabled = std::getenv("EDEN_NX2_WAIT_STATS") != nullptr;
    return enabled;
}

void GpuSplit::NoteWait(const char* what, u32 tag, u64 position) {
    struct Counts {
        u64 count;
        u64 from_end;
        u64 ahead_of_back;
    };
    static std::map<std::pair<std::string, u32>, Counts> counts;
    static auto window = std::chrono::steady_clock::now();
    const u64 back = consumed.load(std::memory_order_acquire);
    auto& entry = counts[{what, tag}];
    ++entry.count;
    entry.from_end += write_position - std::min(position, write_position);
    entry.ahead_of_back += position > back ? position - back : 0;
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds{5}) {
        return;
    }
    std::string line;
    for (const auto& [key, value] : counts) {
        line += fmt::format(" | {} {:#x}: {}/s, {} words before the end, {} past the back", key.first,
                            key.second, value.count / 5, value.from_end / value.count,
                            value.ahead_of_back / value.count);
    }
    LOG_INFO(HW_GPU, "Waits:{}", line);
    counts.clear();
    window = now;
}

void GpuSplit::NoteSkipped(DrainReason reason) {
    if (DrainStatsEnabled()) {
        RecordDrain(reason, false, 0);
    }
}

void GpuSplit::SetUnorderedEngine(Tegra::Engines::EngineInterface* engine, bool unordered) {
    const auto end = unordered_engines.end();
    const auto it = std::find(unordered_engines.begin(), end, engine);
    if (unordered && it == end) {
        if (const auto slot = std::find(unordered_engines.begin(), end, nullptr); slot != end) {
            *slot = engine;
        }
    } else if (!unordered && it != end) {
        *it = nullptr;
    }
    if (engine == front_engine) {
        front_engine_unordered = unordered && std::find(unordered_engines.begin(), end, engine) != end;
    }
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    // Reserve first: it may flush pending writes, which select their engine.
    u64* const entry = Reserve(2);
    front_engine = engine;
    front_engine_unordered =
        engine != nullptr &&
        std::find(unordered_engines.begin(), unordered_engines.end(), engine) != unordered_engines.end();
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
    MarkIfOrdered();
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
}

void GpuSplit::StallWatch(std::stop_token stop_token) {
    // Any one entry that holds the back for 30 ms or more is a stutter: say which, and for how
    // long once it is done. The back only stores its position per entry for this.
    Common::SetCurrentThreadName("GPUStallWatch");
    u64 watched = ~u64{};
    u32 ticks = 0;
    while (!stop_token.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
        const u64 position = executing.load(std::memory_order_relaxed);
        if (position != ~u64{} && position == watched) {
            if (++ticks == 6) {
                ReportStall(&ring[position & RingMask], 30'000'000);
            }
            continue;
        }
        if (ticks >= 6) {
            LOG_WARNING(HW_GPU, "GPU back stall ended after about {} ms", ticks * 5);
        }
        watched = position;
        ticks = 0;
    }
}

void GpuSplit::ReportStall(const u64* entry, u64 took_ns) {
    static u32 reported{};
    if (reported++ >= 200) {
        return;
    }
    const u64 head = entry[0];
    const auto kind = static_cast<Kind>(head & 0xf);
    static constexpr std::array kind_names{"wrap", "engine", "method", "multi", "call", "state", "run"};
    // Read once: the back may change it meanwhile (the watchdog runs beside it).
    const Tegra::Engines::EngineInterface* const engine = back_engine;
    const char* engine_name = "no engine";
    if (engine != nullptr) {
#if defined(__cpp_rtti) || defined(_CPPRTTI) || defined(__GXX_RTTI)
        try {
            engine_name = typeid(*engine).name();
        } catch (...) {
            engine_name = "engine";
        }
#else
        engine_name = "engine";
#endif
    }
    const u64 detail = (head >> 8) & 0xffff;
    if (kind == Kind::Call) {
        LOG_WARNING(HW_GPU, "GPU back stall: over {} ms in a deferred job: {}", took_ns / 1'000'000,
                    reinterpret_cast<const char*>(entry[2]));
        return;
    }
    LOG_WARNING(HW_GPU, "GPU back stall: over {} ms in a {} entry ({} {:#x})", took_ns / 1'000'000,
                static_cast<size_t>(kind) < kind_names.size() ? kind_names[static_cast<size_t>(kind)]
                                                             : "?",
                engine_name, detail);
}

size_t GpuSplit::Execute(const u64* entry) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        // State writes wait in the method sink for the next method with side effects, macro
        // writes too where the engine allows (sink_direct_state): most writes a frame are
        // macro state writes, and CallMethod for each was a large part of the back thread.
        const bool state = !engine->execution_mask[method];
        if ((head & Direct) != 0 && !(state && engine->sink_direct_state)) {
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (state) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::State: {
        const size_t count = static_cast<size_t>((head >> 8) & 0xffffff);
        if (back_engine == nullptr) [[unlikely]] {
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_ERROR(HW_GPU, "GPU split: state writes without an engine, dropped");
            }
            return 1 + count;
        }
        auto& sink = back_engine->method_sink;
        for (size_t i = 1; i <= count; ++i) {
            sink.emplace_back(static_cast<u32>(entry[i] & 0xffff), static_cast<u32>(entry[i] >> 32));
        }
        return 1 + count;
    }
    case Kind::Run: {
        const size_t count = static_cast<size_t>((head >> 8) & 0xffffff);
        auto* const engine = back_engine;
        engine->ConsumeSink(*system);
        engine->CallMethodPairs(*system, entry + 1, static_cast<u32>(count));
        return 1 + count;
    }
    case Kind::Call: {
        Nx2Profile::Scope deferred_scope{Nx2Profile::Zone::Deferred};
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 3);
        return 3 + payload;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPUBack");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    Common::PinCurrentThreadToDedicatedCore(0);
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position; ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                if (available == read_position && !stop_token.stop_requested()) {
                    Common::Nx2Trace::Scope trace{"split", "back idle"};
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        g_back_busy.store(true, std::memory_order_relaxed);
        Common::Nx2Trace::Scope trace{"split", "back runs"};
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            executing.store(read_position, std::memory_order_relaxed); // For StallWatch
            const size_t size = Execute(&ring[index]);
            read_position += size != 0 ? size : RingWords - index;
            // Hand ring space back now and then while working through a long batch.
            if (read_position - last_store >= 1024 && read_position != available) {
                consumed.store(read_position, std::memory_order_release);
                last_store = read_position;
            }
        }
        executing.store(~u64{}, std::memory_order_relaxed);
        g_back_busy.store(false, std::memory_order_release);
        consumed.store(read_position, std::memory_order_release);
    }
}

} // namespace VideoCommon
