#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t scratch_cb_id = get_named_compile_time_arg_val("scratch_cb_id");
    constexpr bool repeat_height = get_named_compile_time_arg_val("repeat_height") != 0;
    constexpr uint32_t repeats = get_named_compile_time_arg_val("repeats");
    constexpr uint32_t element_bytes = get_named_compile_time_arg_val("element_bytes");
    constexpr uint32_t tile_elements = 32 * 32;
    constexpr uint32_t tile_bytes = tile_elements * element_bytes;
    constexpr uint32_t elements_per_word = sizeof(uint32_t) / element_bytes;
    constexpr uint32_t words_per_row = 16 / elements_per_word;
    constexpr bool aligned_width_repeat = 16 % repeats == 0 && repeats <= 8;
    static_assert(repeats >= 2);
    static_assert(element_bytes == 2 || element_bytes == 4);

    const uint32_t source_address = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);
    const uint32_t input_height_tiles = get_arg_val<uint32_t>(3);
    const uint32_t input_width_tiles = get_arg_val<uint32_t>(4);
    const uint32_t output_height = get_arg_val<uint32_t>(5);
    const uint32_t output_width = get_arg_val<uint32_t>(6);
    const uint32_t output_height_tiles = (output_height - 1) / 32 + 1;
    const uint32_t output_width_tiles = (output_width - 1) / 32 + 1;
    const uint32_t input_plane_tiles = input_height_tiles * input_width_tiles;
    const uint32_t output_plane_tiles = output_height_tiles * output_width_tiles;

    constexpr auto source_args = TensorAccessorArgs<0>();
    const auto source = TensorAccessor(source_args, source_address, source_args.get_aligned_page_size());
    Noc noc;
    CircularBuffer output_cb(output_cb_id);
    CircularBuffer scratch_cb(scratch_cb_id);
    scratch_cb.reserve_back(1);
    CoreLocalMem<volatile uint16_t> source_halfwords(scratch_cb.get_write_ptr());
    CoreLocalMem<volatile uint32_t> source_words(scratch_cb.get_write_ptr());
    uint32_t cached_source_page = UINT32_MAX;

    for (uint32_t page_offset = 0; page_offset < num_pages; ++page_offset) {
        const uint32_t output_page = start_page + page_offset;
        const uint32_t plane = output_page / output_plane_tiles;
        const uint32_t page_in_plane = output_page % output_plane_tiles;
        const uint32_t output_row_start = (page_in_plane / output_width_tiles) * 32;
        const uint32_t output_col_start = (page_in_plane % output_width_tiles) * 32;
        const uint32_t valid_rows = output_height - output_row_start < 32 ? output_height - output_row_start : 32;
        const uint32_t valid_cols = output_width - output_col_start < 32 ? output_width - output_col_start : 32;
        const uint32_t source_tile_row = (repeat_height ? output_row_start / repeats : output_row_start) / 32;
        const uint32_t source_tile_col = (repeat_height ? output_col_start : output_col_start / repeats) / 32;
        const uint32_t source_page = plane * input_plane_tiles + source_tile_row * input_width_tiles + source_tile_col;

        if (cached_source_page != source_page) {
            noc.async_read(source, scratch_cb, tile_bytes, {.page_id = source_page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cached_source_page = source_page;
        }

        uint32_t col_offsets[32];
        if constexpr (!repeat_height && !aligned_width_repeat) {
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t local_col = col < valid_cols ? ((output_col_start + col) / repeats) % 32 : 0;
                col_offsets[col] = (local_col / 16) * 256 + local_col % 16;
            }
        }

        output_cb.reserve_back(1);
        CoreLocalMem<volatile uint32_t> destination(output_cb.get_write_ptr());
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t face_col_start = (face % 2) * 16;
            const uint32_t face_cols = valid_cols > face_col_start ? valid_cols - face_col_start : 0;
            uint32_t face_row = 0;
            while (face_row < 16) {
                const uint32_t row = (face / 2) * 16 + face_row;
                uint32_t row_words[words_per_row] = {};
                uint32_t row_copies = 16 - face_row;
                if (row < valid_rows && face_cols != 0) {
                    row_copies = 1;
                    if constexpr (repeat_height) {
                        const uint32_t output_row = output_row_start + row;
                        const uint32_t source_row = (output_row / repeats) % 32;
                        const uint32_t source_offset =
                            ((source_row / 16) * 512 + (face % 2) * 256 + (source_row % 16) * 16) / elements_per_word;
                        const uint32_t remaining_copies = repeats - output_row % repeats;
                        const uint32_t remaining_rows =
                            valid_rows - row < 16 - face_row ? valid_rows - row : 16 - face_row;
                        row_copies = remaining_copies < remaining_rows ? remaining_copies : remaining_rows;
#pragma GCC unroll 16
                        for (uint32_t word = 0; word < words_per_row; ++word) {
                            row_words[word] = source_words[source_offset + word];
                        }
                    } else if constexpr (aligned_width_repeat) {
                        const uint32_t source_col = ((output_col_start + face_col_start) / repeats) % 32;
                        const uint32_t source_offset =
                            ((row / 16) * 512 + (row % 16) * 16 + (source_col / 16) * 256 + source_col % 16) /
                            elements_per_word;
                        constexpr uint32_t input_words_per_row = words_per_row / repeats;
                        uint32_t input_words[words_per_row];
#pragma GCC unroll 8
                        for (uint32_t word = 0; word < input_words_per_row; ++word) {
                            input_words[word] = source_words[source_offset + word];
                        }
#pragma GCC unroll 8
                        for (uint32_t word = 0; word < input_words_per_row; ++word) {
                            const uint32_t value = input_words[word];
                            if constexpr (element_bytes == 2) {
                                const uint32_t lower = (value & 0xFFFFU) | (value << 16);
                                const uint32_t upper = (value >> 16) | (value & 0xFFFF0000U);
#pragma GCC unroll 4
                                for (uint32_t copy = 0; copy < repeats / 2; ++copy) {
                                    row_words[word * repeats + copy] = lower;
                                    row_words[word * repeats + repeats / 2 + copy] = upper;
                                }
                            } else {
#pragma GCC unroll 8
                                for (uint32_t copy = 0; copy < repeats; ++copy) {
                                    row_words[word * repeats + copy] = value;
                                }
                            }
                        }
                    } else {
                        const uint32_t source_row_offset = (row / 16) * 512 + (row % 16) * 16;
#pragma GCC unroll 16
                        for (uint32_t word = 0; word < words_per_row; ++word) {
                            const uint32_t col = face_col_start + word * elements_per_word;
                            const uint32_t source_offset = source_row_offset + col_offsets[col];
                            if constexpr (element_bytes == 2) {
                                row_words[word] =
                                    source_halfwords[source_offset] |
                                    (static_cast<uint32_t>(source_halfwords[source_row_offset + col_offsets[col + 1]])
                                     << 16);
                            } else {
                                row_words[word] = source_words[source_offset];
                            }
                        }
                    }
                    if (face_cols < 16) {
#pragma GCC unroll 16
                        for (uint32_t word = 0; word < words_per_row; ++word) {
                            if (word * elements_per_word >= face_cols) {
                                row_words[word] = 0;
                            }
                        }
                        if constexpr (element_bytes == 2) {
                            if (face_cols % 2 != 0) {
                                row_words[face_cols / 2] &= 0xFFFFU;
                            }
                        }
                    }
                }
                uint32_t destination_offset = (face * 16 + face_row) * words_per_row;
                for (uint32_t copy = 0; copy < row_copies; ++copy) {
#pragma GCC unroll 16
                    for (uint32_t word = 0; word < words_per_row; ++word) {
                        destination[destination_offset + word] = row_words[word];
                    }
                    destination_offset += words_per_row;
                }
                face_row += row_copies;
            }
        }
        output_cb.push_back(1);
    }
}
