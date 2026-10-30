// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_thread.h"
#include "core/memory.h"

namespace Kernel::Svc {

// Logs a guest wait of a second or more that ran into its timeout, with the caller's frame chain,
// to find services that never signal what the game waits for (it stalls for the whole timeout).
inline void LogLongTimedOutWait(Core::System& system, const char* kind, s64 timeout_ns, u64 key) {
    if (timeout_ns < 1'000'000'000) {
        return;
    }
    auto& thread = GetCurrentThread(system.Kernel());
    auto& memory = GetCurrentMemory(system.Kernel());
    const auto& context = thread.GetContext();
    std::string chain = fmt::format("{:#x}", context.lr);
    u64 fp = context.fp;
    for (int depth = 0; depth < 12 && fp != 0 && memory.IsValidVirtualAddressRange(fp, 16); ++depth) {
        chain += fmt::format(",{:#x}", memory.Read64(fp + 8));
        const u64 next = memory.Read64(fp);
        if (next <= fp) {
            break;
        }
        fp = next;
    }
    LOG_WARNING(Kernel_SVC, "{} timed out after {} ms: thread {} key={:#x} chain={}", kind,
                timeout_ns / 1'000'000, thread.GetThreadId(), key, chain);
}

} // namespace Kernel::Svc
