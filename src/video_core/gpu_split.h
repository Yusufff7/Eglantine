// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <typeinfo>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
        MarkIfOrdered();
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment);
    /// A state write (not in the engine's execution_mask) for an engine that sinks them
    /// (EngineInterface::sink_direct_state): collected with the ones around it and sent as one
    /// entry, which the back appends to the engine's method sink.
    void State(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        if (run_size != 0) [[unlikely]] {
            FlushRun();
        }
        if (engine != batch_engine || batch_size == batch.size()) [[unlikely]] {
            if (batch_size != 0) {
                FlushState();
            }
            batch_engine = engine;
        }
        batch[batch_size++] = u64{method & 0xffff} | (u64{argument} << 32);
    }

    /// A write with side effects that may be sent with the next ones of its `group` as one
    /// run (EngineInterface::CallMethodPairs): constant buffer data, a word at a time from macros
    /// across the data methods.
    void RunWord(Tegra::Engines::EngineInterface* engine, u32 group, u32 method, u32 argument) {
        if (batch_size != 0) [[unlikely]] {
            FlushState();
        }
        if (engine != run_engine || group != run_group || run_size == run.size()) [[unlikely]] {
            if (run_size != 0) {
                FlushRun();
            }
            run_engine = engine;
            run_group = group;
        }
        run[run_size++] = u64{method & 0xffff} | (u64{argument} << 32);
    }

    /// Whether a run (RunWord) for `engine` is being collected.
    [[nodiscard]] bool RunPending(Tegra::Engines::EngineInterface* engine) const {
        return run_size != 0 && run_engine == engine;
    }

    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
        MarkIfOrdered();
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// Deferred work counts as ordered (see MarkOrdered) unless `ordered` is false.
    template <typename Func>
    void Defer(const Func& func, bool ordered = true) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(3 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        // The job's type (its lambda's, naming the function it was made in), for StallWatch.
#if defined(__cpp_rtti) || defined(_CPPRTTI) || defined(__GXX_RTTI)
        entry[2] = reinterpret_cast<u64>(typeid(Func).name());
#else
        entry[2] = reinterpret_cast<u64>(JobName<Func>());
#endif
        std::memcpy(entry + 3, &func, sizeof(Func));
        Commit(3 + payload);
        if (ordered) {
            MarkOrdered();
        }
    }
    void DeferFunction(std::function<void()>&& func);

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Why the front waits for the back (EDEN_NX2_DRAIN_STATS=1 logs counts and wait times).
    enum class DrainReason : u32 {
        Other,
        Acquire,       ///< Semaphore acquire check, up to the last ordered method
        AcquireFull,   ///< Semaphore acquire check with the back idle
        AcquireQueued, ///< Semaphore acquire met by a release still queued (no wait)
        AcquireMet,    ///< GEQ semaphore acquire met in memory already (no wait)
        AcquireResume, ///< Retrying a suspended acquire
        Puller,        ///< Other puller semaphore methods
        DmaRead,       ///< Macro MME_DMA_READ, up to the last ordered method
        DmaReadFull,   ///< Macro MME_DMA_READ of memory the back's caches hold
        SyncMemory,    ///< Command list sync (Sync Memory Operations)
        FrontInit,     ///< Engine front state taken over
        Count,
    };

    /// Makes what was sent visible to the back.
    void Publish();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainReason reason = DrainReason::Other);
    /// Waits until the back executed everything sent before `position` (a write position).
    void DrainTo(u64 position, DrainReason reason);
    /// Counts a check answered without waiting (EDEN_NX2_DRAIN_STATS).
    void NoteSkipped(DrainReason reason);

    /// Records that what was sent so far may write memory a later read on the front must see:
    /// a sync point (wait for idle, semaphores), memory writes ordered with the 3D engine's
    /// macros (inline uploads, reports, other engines' work), deferred work.
    void MarkOrdered(u32 tag = 0) {
        ordered_position = write_position;
        write_ordered_position = write_position;
        ordered_tag = tag;
        write_ordered_tag = tag;
    }
    /// A sync point that writes no memory itself (wait for idle, cache invalidations): work
    /// before it may have written what a read after it wants, but only through the caches the
    /// back fills (checked after the read) or memory writes MarkOrdered recorded.
    void MarkBarrier(u32 tag) {
        ordered_position = write_position;
        ordered_tag = tag;
    }
    /// Where the back must be for the front to see the memory writes sent so far (MarkOrdered).
    [[nodiscard]] u64 WriteOrderedPosition() const {
        return write_ordered_position;
    }
    /// What called MarkOrdered last (debugging): 0 deferred work or the caller's own tag,
    /// 1 another engine's method.
    [[nodiscard]] u32 OrderedTag() const {
        return ordered_tag;
    }
    [[nodiscard]] u32 WriteOrderedTag() const {
        return write_ordered_tag;
    }
    /// EDEN_NX2_WAIT_STATS=1 (debugging, front thread): counts `what` by `tag` with how far
    /// `position` is from the last word sent and from where the back is; logs every 5 s.
    static bool WaitStatsEnabled();
    void NoteWait(const char* what, u32 tag, u64 position);

    /// Front-side counts for debugging: words sent, and words the back has not run yet.
    [[nodiscard]] u64 WritePosition() const {
        return write_position;
    }
    [[nodiscard]] u64 ConsumedPosition() const {
        return consumed.load(std::memory_order_acquire);
    }
    /// Where the back must be for the front to read memory the GPU wrote in order (MME DMA reads
    /// only see work that finished, which the game orders with the methods MarkOrdered records).
    [[nodiscard]] u64 OrderedPosition() const {
        return ordered_position;
    }
    /// Semaphore releases sent to the back (front thread only), by device address: an acquire
    /// behind one on the same channel runs after it, so a release still queued that meets the
    /// acquire answers it without waiting for the back.
    void RecordRelease(u64 device_address, u32 payload) {
        auto& entry = releases[release_next++ % releases.size()];
        entry = {device_address, payload, write_position};
    }
    /// The payload of the last release sent to `device_address`, if the back has not run it.
    [[nodiscard]] bool QueuedRelease(u64 device_address, u32& payload) const {
        for (size_t i = 1; i <= releases.size(); ++i) {
            const auto& entry = releases[(release_next - i) % releases.size()];
            if (entry.address == device_address && entry.position != 0) {
                if (entry.position <= consumed.load(std::memory_order_acquire)) {
                    return false;
                }
                payload = entry.payload;
                return true;
            }
        }
        return false;
    }

    /// The write position after the last release sent to the 16 bytes at
    /// [device_begin, device_end) (0 if none of the recorded ones).
    [[nodiscard]] u64 LastReleaseIn(u64 device_begin, u64 device_end) const {
        u64 position = 0;
        for (const auto& entry : releases) {
            if (entry.position > position && entry.address < device_end &&
                device_begin < entry.address + 16) {
                position = entry.position;
            }
        }
        return position;
    }

    /// Debugging: 0 no release recorded for the address, 1 the last one ran already, 2 queued.
    [[nodiscard]] u32 KnownRelease(u64 device_address, u32& payload) const {
        for (size_t i = 1; i <= releases.size(); ++i) {
            const auto& entry = releases[(release_next - i) % releases.size()];
            if (entry.address == device_address && entry.position != 0) {
                payload = entry.payload;
                return entry.position <= consumed.load(std::memory_order_acquire) ? 1 : 2;
            }
        }
        return 0;
    }

    /// The 3D engines mark their memory-writing methods themselves; methods for any other engine
    /// count as ordered.
    void SetUnorderedEngine(Tegra::Engines::EngineInterface* engine, bool unordered);

private:
    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, word 2 type name, then the payload
        State,  ///< Bits 8-31 count; then a word per state write: method, argument << 32
        Run,    ///< Bits 8-31 count; then a word per write, as State (CallMethodPairs)
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Without RTTI (GCC builds use -fno-rtti), the signature naming Func stands in for typeid.
    template <typename Func>
    static const char* JobName() {
#if defined(_MSC_VER)
        return __FUNCSIG__;
#else
        return __PRETTY_FUNCTION__;
#endif
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        if (batch_size != 0) [[unlikely]] {
            FlushState(); // State writes go before what follows them
        }
        if (run_size != 0) [[unlikely]] {
            FlushRun();
        }
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        write_position += words;
        if (write_position - published_local >= publish_every) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        // Pending state writes and runs go first: flushing them selects their own engine.
        if ((batch_size | run_size) != 0) [[unlikely]] {
            if (batch_size != 0) {
                FlushState();
            }
            if (run_size != 0) {
                FlushRun();
            }
        }
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    void FlushState();
    void FlushRun();
    void MarkIfOrdered() {
        if (!front_engine_unordered) {
            ordered_position = write_position;
            write_ordered_position = write_position;
            ordered_tag = 1;
            write_ordered_tag = 1;
        }
    }

    void BackLoop(std::stop_token stop_token);
    size_t Execute(const u64* entry);
    void StallWatch(std::stop_token stop_token);
    void ReportStall(const u64* entry, u64 took_ns);

    Core::System* system{};
    bool active{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};
    std::array<u64, 511> run{};
    size_t run_size{};
    u32 run_group{};
    Tegra::Engines::EngineInterface* run_engine{};
    std::array<u64, 255> batch{};
    size_t batch_size{};
    Tegra::Engines::EngineInterface* batch_engine{};
    bool front_engine_unordered{};
    u64 ordered_position{};
    u64 write_ordered_position{};
    u32 write_ordered_tag{};
    struct QueuedReleaseEntry {
        u64 address;
        u32 payload;
        u64 position;
    };
    std::array<QueuedReleaseEntry, 64> releases{};
    size_t release_next{};
    u32 ordered_tag{};
    std::array<Tegra::Engines::EngineInterface*, 8> unordered_engines{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};

    static GpuSplit instance;

    alignas(64) std::atomic<u64> published{};
    alignas(64) std::atomic<u64> consumed{};
    alignas(64) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    std::jthread back_thread;
    std::atomic<u64> executing{~u64{}}; ///< The entry the back runs (StallWatch)
    std::jthread stall_thread;
};

} // namespace VideoCommon
