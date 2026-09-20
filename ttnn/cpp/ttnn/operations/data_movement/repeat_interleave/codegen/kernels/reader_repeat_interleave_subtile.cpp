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
            cached_source_page = source_page;
        }
        noc.async_read_barrier();

        uint32_t row_offsets[32];
        uint32_t col_offsets[32];
        for (uint32_t row = 0; row < valid_rows; ++row) {
            const uint32_t source_row = repeat_height ? (output_row_start + row) / repeats : output_row_start + row;
            const uint32_t local_row = source_row % 32;
            row_offsets[row] = (local_row / 16) * 512 + (local_row % 16) * 16;
        }
        for (uint32_t col = 0; col < valid_cols; ++col) {
            const uint32_t source_col = repeat_height ? output_col_start + col : (output_col_start + col) / repeats;
            const uint32_t local_col = source_col % 32;
            col_offsets[col] = (local_col / 16) * 256 + local_col % 16;
        }

        output_cb.reserve_back(1);
        CoreLocalMem<volatile uint32_t> destination(output_cb.get_write_ptr());
        uint32_t word_index = 0;
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t face_row = 0; face_row < 16; ++face_row) {
                const uint32_t row = (face / 2) * 16 + face_row;
                for (uint32_t face_col = 0; face_col < 16; face_col += elements_per_word) {
                    const uint32_t col = (face % 2) * 16 + face_col;
                    uint32_t word = 0;
                    if (row < valid_rows && col < valid_cols) {
                        const uint32_t source_offset = row_offsets[row] + col_offsets[col];
                        if constexpr (element_bytes == 2) {
                            word = source_halfwords[source_offset];
                            if (col + 1 < valid_cols) {
                                word |= static_cast<uint32_t>(source_halfwords[row_offsets[row] + col_offsets[col + 1]])
                                        << 16;
                            }
                        } else {
                            word = source_words[source_offset];
                        }
                    }
                    destination[word_index++] = word;
                }
            }
        }
        output_cb.push_back(1);
    }
}
