// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "video_core/background_destroyer.h"

namespace VideoCommon {

/// Container to push objects to be destroyed a few ticks in the future
template <typename T, size_t TICKS_TO_DESTROY>
class DelayedDestructionRing {
public:
    /// With `background`, the expired objects are destroyed by the BackgroundDestroyer; objects
    /// with ReleaseFrontState() drop what only this thread may touch first.
    void Tick(bool background = false) {
        index = (index + 1) % TICKS_TO_DESTROY;
        auto& expired = elements[index];
        if (!background || expired.empty() || !BackgroundDestroyer::Enabled()) {
            expired.clear();
            return;
        }
        if constexpr (requires(T& object) { object.ReleaseFrontState(); }) {
            for (T& object : expired) {
                object.ReleaseFrontState();
            }
        }
        BackgroundDestroyer::Instance().Destroy(std::exchange(expired, {}));
    }

    void Push(T&& object) {
        elements[index].push_back(std::move(object));
    }

private:
    size_t index = 0;
    std::array<std::vector<T>, TICKS_TO_DESTROY> elements;
};

} // namespace VideoCommon
