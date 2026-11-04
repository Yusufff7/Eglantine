// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "core/hle/kernel/k_scoped_lock.h"

namespace Kernel {

class KernelCore;

class KThread;

class KLightLock {
public:
    explicit KLightLock(KernelCore& kernel) : m_kernel{kernel} {}

    void Lock();

    /// Takes the lock if it is free, without waiting: for host threads that must not block on
    /// a guest kernel lock (the GPU thread, whose waits can depend on threads holding it).
    bool TryLock();

    void Unlock();

    bool LockSlowPath(uintptr_t owner, uintptr_t cur_thread);

    void UnlockSlowPath(uintptr_t cur_thread);

    bool IsLocked() const {
        return m_tag.load() != 0;
    }

    bool IsLockedByCurrentThread() const;

    /// The thread holding the lock (debugging: who keeps it), or null.
    KThread* GetOwner() const {
        return reinterpret_cast<KThread*>(m_tag.load(std::memory_order_relaxed) & ~uintptr_t{1});
    }

private:
    std::atomic<uintptr_t> m_tag{};
    KernelCore& m_kernel;
};

using KScopedLightLock = KScopedLock<KLightLock>;

} // namespace Kernel
