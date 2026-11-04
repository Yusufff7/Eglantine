// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <thread>

namespace VideoCommon::Nx2Sampler {

/// EDEN_NX2_SAMPLE=<first>:<last> (presents): samples `thread`'s call stack every millisecond
/// between those presents, then logs the hottest functions, by own time and with callees (finding
/// where a CPU-bound thread's time goes). EDEN_NX2_SAMPLE_THREAD picks the thread by `name`
/// ("back" unless set). Windows only; does nothing otherwise or when unset.
void Start(std::thread::native_handle_type thread, const char* name);

/// Start for the calling thread.
void StartForCurrentThread(const char* name);

} // namespace VideoCommon::Nx2Sampler
