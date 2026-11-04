// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "common/alignment.h"
#include "common/scope_exit.h"
#include "common/div_ceil.h"
#include "common/scratch_buffer.h"
#include "common/thread_worker.h"
#include "video_core/engines/sw_blitter/blitter.h"
#include "video_core/nx2.h"
#include "video_core/engines/sw_blitter/converter.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace Tegra {
class MemoryManager;
}

using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

namespace Tegra::Engines::Blitter {

using namespace Texture;

namespace {

constexpr size_t ir_components = 4;

void NearestNeighbor(std::span<const u8> input, std::span<u8> output, u32 src_width, u32 src_height,
                     u32 dst_width, u32 dst_height, size_t bpp) {
    const size_t dx_du = std::llround((static_cast<f64>(src_width) / dst_width) * (1ULL << 32));
    const size_t dy_dv = std::llround((static_cast<f64>(src_height) / dst_height) * (1ULL << 32));
    size_t src_y = 0;
    for (u32 y = 0; y < dst_height; y++) {
        size_t src_x = 0;
        for (u32 x = 0; x < dst_width; x++) {
            const size_t read_from = ((src_y * src_width + src_x) >> 32) * bpp;
            const size_t write_to = (y * dst_width + x) * bpp;

            std::memcpy(&output[write_to], &input[read_from], bpp);
            src_x += dx_du;
        }
        src_y += dy_dv;
    }
}

void NearestNeighborFast(std::span<const f32> input, std::span<f32> output, u32 src_width,
                         u32 src_height, u32 dst_width, u32 dst_height) {
    const size_t dx_du = std::llround((static_cast<f64>(src_width) / dst_width) * (1ULL << 32));
    const size_t dy_dv = std::llround((static_cast<f64>(src_height) / dst_height) * (1ULL << 32));
    size_t src_y = 0;
    for (u32 y = 0; y < dst_height; y++) {
        size_t src_x = 0;
        for (u32 x = 0; x < dst_width; x++) {
            const size_t read_from = ((src_y * src_width + src_x) >> 32) * ir_components;
            const size_t write_to = (y * dst_width + x) * ir_components;

            std::memcpy(&output[write_to], &input[read_from], sizeof(f32) * ir_components);
            src_x += dx_du;
        }
        src_y += dy_dv;
    }
}

void Bilinear(std::span<const f32> input, std::span<f32> output, size_t src_width,
              size_t src_height, size_t dst_width, size_t dst_height) {
    const auto bilinear_sample = [](std::span<const f32> x0_y0, std::span<const f32> x1_y0,
                                    std::span<const f32> x0_y1, std::span<const f32> x1_y1,
                                    f32 weight_x, f32 weight_y) {
        std::array<f32, ir_components> result{};
        for (size_t i = 0; i < ir_components; i++) {
            const f32 a = std::lerp(x0_y0[i], x1_y0[i], weight_x);
            const f32 b = std::lerp(x0_y1[i], x1_y1[i], weight_x);
            result[i] = std::lerp(a, b, weight_y);
        }
        return result;
    };
    const f32 dx_du =
        dst_width > 1 ? static_cast<f32>(src_width - 1) / static_cast<f32>(dst_width - 1) : 0.f;
    const f32 dy_dv =
        dst_height > 1 ? static_cast<f32>(src_height - 1) / static_cast<f32>(dst_height - 1) : 0.f;
    for (u32 y = 0; y < dst_height; y++) {
        for (u32 x = 0; x < dst_width; x++) {
            const f32 x_low = std::floor(static_cast<f32>(x) * dx_du);
            const f32 y_low = std::floor(static_cast<f32>(y) * dy_dv);
            const f32 x_high = std::ceil(static_cast<f32>(x) * dx_du);
            const f32 y_high = std::ceil(static_cast<f32>(y) * dy_dv);
            const f32 weight_x = (static_cast<f32>(x) * dx_du) - x_low;
            const f32 weight_y = (static_cast<f32>(y) * dy_dv) - y_low;

            const auto read_src = [&](f32 in_x, f32 in_y) {
                const size_t read_from =
                    ((static_cast<size_t>(in_x) * src_width + static_cast<size_t>(in_y)) >> 32) *
                    ir_components;
                return std::span<const f32>(&input[read_from], ir_components);
            };

            auto x0_y0 = read_src(x_low, y_low);
            auto x1_y0 = read_src(x_high, y_low);
            auto x0_y1 = read_src(x_low, y_high);
            auto x1_y1 = read_src(x_high, y_high);

            const auto result = bilinear_sample(x0_y0, x1_y0, x0_y1, x1_y1, weight_x, weight_y);

            const size_t write_to = (y * dst_width + x) * ir_components;

            std::memcpy(&output[write_to], &result, sizeof(f32) * ir_components);
        }
    }
}

template <bool unpack>
void ProcessPitchLinear(std::span<const u8> input, std::span<u8> output, size_t extent_x,
                        size_t extent_y, u32 pitch, u32 x0, u32 y0, size_t bpp) {
    const size_t base_offset = x0 * bpp;
    const size_t copy_size = extent_x * bpp;
    for (size_t y = 0; y < extent_y; y++) {
        const size_t first_offset = (y + y0) * pitch + base_offset;
        const size_t second_offset = y * extent_x * bpp;
        u8* write_to = unpack ? &output[first_offset] : &output[second_offset];
        const u8* read_from = unpack ? &input[second_offset] : &input[first_offset];
        std::memcpy(write_to, read_from, copy_size);
    }
}

} // namespace

struct SoftwareBlitEngine::BlitEngineImpl {
    Common::ScratchBuffer<u8> tmp_buffer;
    Common::ScratchBuffer<u8> dst_tmp_buffer; ///< The destination while the source is in use

    Common::ScratchBuffer<u8> src_buffer;
    Common::ScratchBuffer<u8> dst_buffer;
    Common::ScratchBuffer<f32> intermediate_src;
    Common::ScratchBuffer<f32> intermediate_dst;
    ConverterFactory converter_factory;
};

SoftwareBlitEngine::SoftwareBlitEngine(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_} {
    impl = std::make_unique<BlitEngineImpl>();
}

SoftwareBlitEngine::~SoftwareBlitEngine() = default;

bool SoftwareBlitEngine::Blit(Fermi2D::Surface& src, Fermi2D::Surface& dst,
                              Fermi2D::Config& config) {
    const auto get_surface_size = [](Fermi2D::Surface& surface, u32 bytes_per_pixel) {
        if (surface.linear == Fermi2D::MemoryLayout::BlockLinear) {
            return CalculateSize(true, bytes_per_pixel, surface.width, surface.height,
                                 surface.depth, surface.block_height, surface.block_depth);
        }
        return static_cast<size_t>(surface.pitch * surface.height);
    };

    const u32 src_extent_x = config.src_x1 - config.src_x0;
    const u32 src_extent_y = config.src_y1 - config.src_y0;

    const u32 dst_extent_x = config.dst_x1 - config.dst_x0;
    const u32 dst_extent_y = config.dst_y1 - config.dst_y0;
    const auto src_bytes_per_pixel = BytesPerBlock(PixelFormatFromRenderTargetFormat(src.format));
    const auto dst_bytes_per_pixel = BytesPerBlock(PixelFormatFromRenderTargetFormat(dst.format));
    const size_t src_size = get_surface_size(src, src_bytes_per_pixel);

    // Slow blits (a frame or more) are stutters: log where the time went.
    const auto blit_start = std::chrono::steady_clock::now();
    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::SafeRead> tmp_buffer(
        memory_manager, src.Address(), src_size, &impl->tmp_buffer);
    const auto read_done = std::chrono::steady_clock::now();
    auto converted = read_done;
    auto unswizzled = read_done;
    std::chrono::steady_clock::duration swizzling{};
    std::chrono::steady_clock::duration mapping{};
    u32 read_rows = 0;
    const char* path = "general";
    SCOPE_EXIT {
        const auto end = std::chrono::steady_clock::now();
        if (end - blit_start > std::chrono::milliseconds{20}) {
            static u32 logged{};
            if (logged++ < 64) {
                const auto ms = [](auto d) {
                    return std::chrono::duration<double, std::milli>(d).count();
                };
                LOG_WARNING(HW_GPU,
                            "Slow software 2D blit ({}): {:.1f} ms ({:.1f} reading the source, {:.1f} "
                            "unswizzling, {:.1f} converting and writing ({:.1f} swizzling, {:.1f} mapping, {} rows read), {:.1f} writing back): src {}x{} bh {} origin {},{} dst {}x{} bh {} origin {},{}; {}x{} "
                            "format {:#x} {} at {:#x} -> {}x{} format {:#x} {} at {:#x}, {} bytes read",
                            path, ms(end - blit_start), ms(read_done - blit_start),
                            ms(unswizzled - read_done), ms(converted - unswizzled), ms(swizzling), ms(mapping), read_rows, ms(end - converted), src.width, src.height, static_cast<u32>(src.block_height),
                            config.src_x0, config.src_y0, dst.width, dst.height,
                            static_cast<u32>(dst.block_height), config.dst_x0, config.dst_y0, src_extent_x,
                            src_extent_y, static_cast<u32>(src.format),
                            src.linear == Fermi2D::MemoryLayout::BlockLinear ? "block" : "pitch",
                            src.Address(), dst_extent_x, dst_extent_y,
                            static_cast<u32>(dst.format),
                            dst.linear == Fermi2D::MemoryLayout::BlockLinear ? "block" : "pitch",
                            dst.Address(), src_size);
            }
        }
    };

    // A copy of a whole surface into one of the same size, format and layout moves the same
    // bytes: write them over at once instead of unswizzling and swizzling every pixel (16 MiB
    // RGBA32F textures took 20-90 ms that way, a stutter each).
    const bool same_layout =
        src.format == dst.format && src.linear == dst.linear && src.width == dst.width &&
        src.height == dst.height && src.depth == dst.depth &&
        (src.linear == Fermi2D::MemoryLayout::BlockLinear
             ? src.block_height == dst.block_height && src.block_depth == dst.block_depth
             : src.pitch == dst.pitch);
    const bool whole_surface = config.src_x0 == 0 && config.src_y0 == 0 && config.dst_x0 == 0 &&
                               config.dst_y0 == 0 && src_extent_x == src.width &&
                               src_extent_y == src.height && dst_extent_x == src_extent_x &&
                               dst_extent_y == src_extent_y;
    if (same_layout && whole_surface && src_size == get_surface_size(dst, dst_bytes_per_pixel)) {
        const auto back_start = std::chrono::steady_clock::now();
        Tegra::Nx2BackSparseWrite(dst.Address(), src_size);
        const auto write_start = std::chrono::steady_clock::now();
        memory_manager.WriteBlock(dst.Address(), tmp_buffer.data(), src_size);
        const auto write_end = std::chrono::steady_clock::now();
        if (write_end - back_start > std::chrono::milliseconds{20}) {
            static u32 logged{};
            if (logged++ < 32) {
                LOG_WARNING(HW_GPU, "Whole-surface 2D copy of {} bytes: {:.1f} ms backing, {:.1f} ms writing",
                            src_size,
                            std::chrono::duration<double, std::milli>(write_start - back_start).count(),
                            std::chrono::duration<double, std::milli>(write_end - write_start).count());
            }
        }
        return true;
    }

    // The same rectangle of two tiled surfaces of one shape (title 1: 1023x1023 of 1024x1024
    // RGBA32F, 20-50 ms each through the linear buffers): copy it in place, GOB by GOB.
    const bool same_rect = config.src_x0 == config.dst_x0 && config.src_y0 == config.dst_y0 &&
                           src_extent_x == dst_extent_x && src_extent_y == dst_extent_y;
    if (same_layout && same_rect && src.linear == Fermi2D::MemoryLayout::BlockLinear &&
        src.depth == 1 && std::has_single_bit(src_bytes_per_pixel) && src_bytes_per_pixel <= 16 &&
        src_size == get_surface_size(dst, dst_bytes_per_pixel)) {
        Tegra::Memory::GpuGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::SafeReadWrite>
            dst_memory(memory_manager, dst.Address(), src_size, &impl->dst_tmp_buffer);
        CopySwizzledSubrect(dst_memory, tmp_buffer, src_bytes_per_pixel, src.width, src.height,
                            config.src_x0, config.src_y0, src_extent_x, src_extent_y,
                            src.block_height);
        path = "in place";
        converted = std::chrono::steady_clock::now();
        return true;
    }

    const size_t src_copy_size = src_extent_x * src_extent_y * src_bytes_per_pixel;
    const size_t dst_copy_size = dst_extent_x * dst_extent_y * dst_bytes_per_pixel;

    impl->src_buffer.resize_destructive(src_copy_size);

    const bool no_passthrough =
        src.format != dst.format || src_extent_x != dst_extent_x || src_extent_y != dst_extent_y;

    const auto conversion_phase_same_format = [&]() {
        NearestNeighbor(impl->src_buffer, impl->dst_buffer, src_extent_x, src_extent_y,
                        dst_extent_x, dst_extent_y, dst_bytes_per_pixel);
    };

    const auto conversion_phase_ir = [&]() {
        auto* input_converter = impl->converter_factory.GetFormatConverter(src.format);
        impl->intermediate_src.resize_destructive((src_copy_size / src_bytes_per_pixel) *
                                                  ir_components);
        impl->intermediate_dst.resize_destructive((dst_copy_size / dst_bytes_per_pixel) *
                                                  ir_components);
        input_converter->ConvertTo(impl->src_buffer, impl->intermediate_src);

        if (config.filter != Fermi2D::Filter::Bilinear) {
            NearestNeighborFast(impl->intermediate_src, impl->intermediate_dst, src_extent_x,
                                src_extent_y, dst_extent_x, dst_extent_y);
        } else {
            Bilinear(impl->intermediate_src, impl->intermediate_dst, src_extent_x, src_extent_y,
                     dst_extent_x, dst_extent_y);
        }

        auto* output_converter = impl->converter_factory.GetFormatConverter(dst.format);
        output_converter->ConvertFrom(impl->intermediate_dst, impl->dst_buffer);
    };

    // Do actual Blit

    impl->dst_buffer.resize_destructive(dst_copy_size);
    if (src.linear == Fermi2D::MemoryLayout::BlockLinear && src.depth == 1 &&
        src_copy_size >= (4u << 20)) {
        // Large ones (16-64 MiB RGBA32F, 5-30 ms on one core) in bands of rows on the workers.
        static Common::ThreadWorker workers((std::max)(2u, (std::min)(8u, std::thread::hardware_concurrency() / 4)),
                                            "BlitUnswizzle");
        const u32 pitch = src_extent_x * src_bytes_per_pixel;
        const u32 bands = 8;
        const u32 band_rows = Common::DivCeil(src_extent_y, bands);
        const std::span<u8> output = impl->src_buffer;
        const std::span<const u8> input = tmp_buffer;
        for (u32 first = 0; first < src_extent_y && config.src_y0 + first < src.height;
             first += band_rows) {
            const u32 rows = (std::min)(band_rows, src_extent_y - first);
            workers.QueueWork([=, &src, &config] {
                UnswizzleSubrect(output.subspan(size_t{first} * pitch), input, src_bytes_per_pixel,
                                 src.width, src.height, 1, config.src_x0, config.src_y0 + first,
                                 src_extent_x, rows, src.block_height, src.block_depth, pitch);
            });
        }
        workers.WaitForRequests();
    } else if (src.linear == Fermi2D::MemoryLayout::BlockLinear) {
        UnswizzleSubrect(impl->src_buffer, tmp_buffer, src_bytes_per_pixel, src.width, src.height,
                         src.depth, config.src_x0, config.src_y0, src_extent_x, src_extent_y,
                         src.block_height, src.block_depth, src_extent_x * src_bytes_per_pixel);
    } else {
        ProcessPitchLinear<false>(tmp_buffer, impl->src_buffer, src_extent_x, src_extent_y,
                                  src.pitch, config.src_x0, config.src_y0, src_bytes_per_pixel);
    }

    unswizzled = std::chrono::steady_clock::now();
    // Conversion Phase
    if (no_passthrough) {
        if (src.format != dst.format || config.filter == Fermi2D::Filter::Bilinear) {
            conversion_phase_ir();
        } else {
            conversion_phase_same_format();
        }
    } else {
        impl->dst_buffer.swap(impl->src_buffer);
    }

    // Into a 2D tiled surface: write only the block rows and GOB columns the rectangle covers,
    // without reading back the ones it covers whole. The whole destination was read and written
    // (title 1: 1024x1024 RGBA32F into the corner of 2048x2048, 64 MiB each way, 40-75 ms).
    if (dst.linear == Fermi2D::MemoryLayout::BlockLinear && dst.depth == 1 &&
        std::has_single_bit(dst_bytes_per_pixel) && dst_bytes_per_pixel <= 16 &&
        dst_extent_x > 0 && dst_extent_y > 0 && config.dst_y0 < dst.height) {
        const u32 block_height = dst.block_height;
        const u32 rows_per_block = GOB_SIZE_Y << block_height;
        const u32 gobs_in_x = Common::AlignUpLog2(dst.width * dst_bytes_per_pixel, GOB_SIZE_X_SHIFT) >>
                              GOB_SIZE_X_SHIFT;
        const u64 block_size = u64{gobs_in_x} << (GOB_SIZE_SHIFT + block_height);
        const u32 x_shift = GOB_SIZE_SHIFT + block_height;
        const u32 first_byte = config.dst_x0 * dst_bytes_per_pixel;
        const u32 end_byte = (config.dst_x0 + dst_extent_x) * dst_bytes_per_pixel;
        const u32 first_gob = first_byte >> GOB_SIZE_X_SHIFT;
        const u32 gob_columns = ((end_byte - 1) >> GOB_SIZE_X_SHIFT) - first_gob + 1;
        const u32 columns_width = (gob_columns << GOB_SIZE_X_SHIFT) / dst_bytes_per_pixel;
        const u32 origin_x = config.dst_x0 - (first_gob << GOB_SIZE_X_SHIFT) / dst_bytes_per_pixel;
        const bool whole_columns = (first_byte & (GOB_SIZE_X - 1)) == 0 &&
                                   (end_byte & (GOB_SIZE_X - 1)) == 0;
        const u32 pitch = dst_extent_x * dst_bytes_per_pixel;
        const u32 first_row = config.dst_y0;
        const u32 end_row = (std::min)(config.dst_y0 + dst_extent_y, dst.height);
        // EDEN_NX2_BLIT_VERIFY: also swizzle the whole surface the old way and compare.
        static const bool verify = std::getenv("EDEN_NX2_BLIT_VERIFY") != nullptr;
        std::vector<u8> expected;
        if (verify) {
            expected.resize(get_surface_size(dst, dst_bytes_per_pixel));
            memory_manager.ReadBlockUnsafe(dst.Address(), expected.data(), expected.size());
            SwizzleSubrect(expected, impl->dst_buffer, dst_bytes_per_pixel, dst.width, dst.height,
                           dst.depth, config.dst_x0, config.dst_y0, dst_extent_x, dst_extent_y,
                           dst.block_height, dst.block_depth, pitch);
        }
        for (u32 block_row = first_row / rows_per_block; block_row * rows_per_block < end_row;
             ++block_row) {
            const u32 block_top = block_row * rows_per_block;
            const u32 rows_begin = (std::max)(first_row, block_top);
            const u32 rows_end = (std::min)(end_row, block_top + rows_per_block);
            const GPUVAddr address =
                dst.Address() + block_row * block_size + (u64{first_gob} << x_shift);
            const size_t size = size_t{gob_columns} << x_shift;
            const auto input = std::span<const u8>(impl->dst_buffer)
                                   .subspan(size_t{rows_begin - first_row} * pitch);
            const auto swizzle = [&](std::span<u8> output) {
                const auto swizzle_start = std::chrono::steady_clock::now();
                SCOPE_EXIT {
                    swizzling += std::chrono::steady_clock::now() - swizzle_start;
                };
                SwizzleSubrect(output, input, dst_bytes_per_pixel, columns_width, rows_per_block, 1,
                               origin_x, rows_begin - block_top, dst_extent_x,
                               rows_end - rows_begin, block_height, 0, pitch);
            };
            const auto map_start = std::chrono::steady_clock::now();
            if (whole_columns && rows_end - rows_begin == rows_per_block) {
                Tegra::Memory::GpuGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::SafeWrite>
                    memory(memory_manager, address, size, &impl->tmp_buffer);
                mapping += std::chrono::steady_clock::now() - map_start;
                swizzle(memory);
            } else {
                Tegra::Memory::GpuGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::SafeReadWrite>
                    memory(memory_manager, address, size, &impl->tmp_buffer);
                mapping += std::chrono::steady_clock::now() - map_start;
                ++read_rows;
                swizzle(memory);
            }
        }
        if (verify) {
            std::vector<u8> written(expected.size());
            memory_manager.ReadBlockUnsafe(dst.Address(), written.data(), written.size());
            size_t mismatches = 0;
            for (size_t i = 0; i < written.size(); ++i) {
                mismatches += written[i] != expected[i];
            }
            static u32 logged{};
            if (logged++ < 32) {
                LOG_WARNING(HW_GPU, "2D blit by block rows check: {} of {} bytes differ ({}x{} at {},{} into {}x{} bh {})",
                            mismatches, written.size(), dst_extent_x, dst_extent_y, config.dst_x0,
                            config.dst_y0, dst.width, dst.height, block_height);
            }
        }
        path = "by block rows";
        converted = std::chrono::steady_clock::now();
        return true;
    }

    const size_t dst_size = get_surface_size(dst, dst_bytes_per_pixel);
    Tegra::Memory::GpuGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::SafeReadWrite>
        tmp_buffer2(memory_manager, dst.Address(), dst_size, &impl->tmp_buffer);

    if (dst.linear == Fermi2D::MemoryLayout::BlockLinear) {
        SwizzleSubrect(tmp_buffer2, impl->dst_buffer, dst_bytes_per_pixel, dst.width, dst.height,
                       dst.depth, config.dst_x0, config.dst_y0, dst_extent_x, dst_extent_y,
                       dst.block_height, dst.block_depth, dst_extent_x * dst_bytes_per_pixel);
    } else {
        ProcessPitchLinear<true>(impl->dst_buffer, tmp_buffer2, dst_extent_x, dst_extent_y,
                                 dst.pitch, config.dst_x0, config.dst_y0,
                                 static_cast<size_t>(dst_bytes_per_pixel));
    }
    converted = std::chrono::steady_clock::now();
    return true;
}

} // namespace Tegra::Engines::Blitter
