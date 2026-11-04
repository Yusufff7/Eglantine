// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <limits>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {

enum class EngineTypes : u32 {
    Nv01Timer,
    KeplerCompute,
    Maxwell3D,
    Fermi2D,
    MaxwellDMA,
    KeplerMemory,
};

class EngineInterface {
public:
    virtual ~EngineInterface() = default;

    /// Write the value to the register identified by method.
    virtual void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) = 0;

    /// Write multiple values to the register identified by method.
    virtual void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) = 0;

    /// Writes in a row (method | argument << 32 each), each as CallMethod would make it (the
    /// last is the last call): engines may do them at once where that is the same.
    virtual void CallMethodPairs(Core::System& system, const u64* pairs, u32 count) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, static_cast<u32>(pairs[i] & 0xffff), static_cast<u32>(pairs[i] >> 32),
                       i + 1 == count);
        }
    }

    void ConsumeSink(Core::System& system) {
        if (!method_sink.empty()) {
            ConsumeSinkImpl(system);
        }
    }

    std::bitset<(std::numeric_limits<u16>::max)()> execution_mask{};
    std::vector<std::pair<u32, u32>> method_sink{};
    /// State writes a macro makes may go through method_sink too: the engine's ConsumeSinkImpl
    /// does to them what CallMethod would (Maxwell3D: shadow RAM, same-value skip, dirty flags).
    bool sink_direct_state{};
    GPUVAddr current_dma_segment;
    bool current_dirty{};
protected:
    virtual void ConsumeSinkImpl(Core::System& system) {
        for (auto [method, value] : method_sink) {
            CallMethod(system, method, value, true);
        }
        method_sink.clear();
    }
};

} // namespace Tegra::Engines
