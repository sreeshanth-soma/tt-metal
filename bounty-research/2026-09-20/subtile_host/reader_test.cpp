#include <iostream>
#include <random>

#include "runtime.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/reader_repeat_interleave_subtile.cpp"

namespace subtile_host {

uint64_t checked_elements = 0;
uint64_t core_invocations = 0;
uint64_t page_reads = 0;
uint64_t secondary_reads = 0;
uint64_t source_l1_halfword_reads = 0;
uint64_t source_l1_word_reads = 0;
uint64_t output_l1_word_writes = 0;

uint64_t physical_index(uint32_t plane, uint32_t row, uint32_t col, uint32_t height, uint32_t width) {
    const uint32_t height_tiles = (height + 31) / 32;
    const uint32_t width_tiles = (width + 31) / 32;
    const uint64_t tile = (static_cast<uint64_t>(plane) * height_tiles + row / 32) * width_tiles + col / 32;
    const uint32_t face = ((row % 32) / 16) * 2 + (col % 32) / 16;
    return tile * 1024 + face * 256 + (row % 16) * 16 + col % 16;
}

uint32_t input_bits(uint64_t logical_index, uint32_t seed) {
    constexpr std::array<uint32_t, 12> patterns32 = {
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC23456,
        0x7F812345};
    constexpr std::array<uint32_t, 12> patterns16 = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F81};
    const auto& patterns = element_bytes == 2 ? patterns16 : patterns32;
    const uint64_t adjusted = logical_index + seed;
    const uint32_t generated = adjusted % 17 < patterns.size()
                                   ? patterns[adjusted % 17]
                                   : static_cast<uint32_t>(adjusted * 2654435761ULL + seed * 2246822519ULL);
    return element_bytes == 2 ? generated & 0xFFFF : generated;
}

void set_element(std::vector<uint32_t>& storage, uint64_t index, uint32_t value) {
    if constexpr (element_bytes == 2) {
        const uint32_t shift = (index % 2) * 16;
        storage.at(index / 2) = (storage.at(index / 2) & ~(0xFFFFU << shift)) | (value << shift);
    } else {
        storage.at(index) = value;
    }
}

void run_case(uint32_t planes, uint32_t height, uint32_t width, uint32_t workers, uint32_t seed) {
    const uint32_t input_pages = planes * ((height + 31) / 32) * ((width + 31) / 32);
    const uint32_t output_height = height * (TEST_REPEAT_HEIGHT ? TEST_REPEATS : 1);
    const uint32_t output_width = width * (TEST_REPEAT_HEIGHT ? 1 : TEST_REPEATS);
    const uint32_t output_pages = planes * ((output_height + 31) / 32) * ((output_width + 31) / 32);
    std::vector<uint32_t> input(input_pages * tile_words, 0xDEADDEAD);
    std::vector<uint32_t> actual(output_pages * tile_words, guard_value);
    std::vector<uint32_t> expected(output_pages * tile_words, 0);

    for (uint32_t plane = 0; plane < planes; ++plane) {
        for (uint32_t row = 0; row < height; ++row) {
            for (uint32_t col = 0; col < width; ++col) {
                const uint64_t logical_index = (static_cast<uint64_t>(plane) * height + row) * width + col;
                set_element(input, physical_index(plane, row, col, height, width), input_bits(logical_index, seed));
            }
        }
        for (uint32_t row = 0; row < output_height; ++row) {
            for (uint32_t col = 0; col < output_width; ++col) {
                const uint32_t source_row = TEST_REPEAT_HEIGHT ? row / TEST_REPEATS : row;
                const uint32_t source_col = TEST_REPEAT_HEIGHT ? col : col / TEST_REPEATS;
                const uint64_t logical_index =
                    (static_cast<uint64_t>(plane) * height + source_row) * width + source_col;
                set_element(
                    expected,
                    physical_index(plane, row, col, output_height, output_width),
                    input_bits(logical_index, seed));
            }
        }
    }

    workers = std::min(workers, output_pages);
    uint32_t start_page = 0;
    for (uint32_t worker = 0; worker < workers; ++worker) {
        const uint32_t pages = output_pages / workers + (worker < output_pages % workers);
        Context context(
            {0x10000U + seed * 0x100U,
             pages,
             start_page,
             (height + 31) / 32,
             (width + 31) / 32,
             output_height,
             output_width},
            input,
            actual);
        active_context = &context;
        kernel_main();
        context.check_complete();
        page_reads += context.source_reads;
        secondary_reads += context.second_slot_reads;
        source_l1_halfword_reads += context.source_l1_halfword_reads;
        source_l1_word_reads += context.source_l1_word_reads;
        output_l1_word_writes += context.output_l1_word_writes;
        ++core_invocations;
        start_page += pages;
    }
    active_context = nullptr;
    require(start_page == output_pages, "work partition omitted pages");
    if (actual != expected) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        throw std::runtime_error(
            "bit mismatch planes=" + std::to_string(planes) + " height=" + std::to_string(height) +
            " width=" + std::to_string(width) + " workers=" + std::to_string(workers) +
            " word=" + std::to_string(mismatch.first - actual.begin()));
    }
    checked_elements += static_cast<uint64_t>(output_pages) * 1024;
}

}

int main() {
    try {
        constexpr std::array<uint32_t, 13> extents = {1, 2, 3, 15, 16, 17, 31, 32, 33, 37, 63, 64, 65};
        constexpr std::array<uint32_t, 5> worker_counts = {1, 2, 7, 72, 80};
        uint32_t cases = 0;
        for (const uint32_t height : extents) {
            for (const uint32_t width : extents) {
                const uint32_t planes = cases % 3 + 1;
                subtile_host::run_case(planes, height, width, worker_counts[cases % worker_counts.size()], cases + 1);
                ++cases;
            }
        }
        std::mt19937 generator(1771);
        for (uint32_t sample = 0; sample < 40; ++sample) {
            const uint32_t height = generator() % 95 + 1;
            const uint32_t width = generator() % 95 + 1;
            const uint32_t planes = generator() % 4 + 1;
            const uint32_t workers = worker_counts[generator() % worker_counts.size()];
            subtile_host::run_case(planes, height, width, workers, cases + 1);
            ++cases;
            subtile_host::run_case(planes, height, width, workers, cases + 1);
            ++cases;
        }
        subtile_host::run_case(4, 64, 128, 72, 500);
        subtile_host::run_case(128, 8, 64, 72, 501);
        cases += 2;
        subtile_host::require(subtile_host::secondary_reads == 0, "one source tile must cover each output tile");
        subtile_host::require(
            subtile_host::page_reads < subtile_host::checked_elements / 1024, "source-page cache not exercised");
        std::cout << "{\"cases\":" << cases << ",\"core_invocations\":" << subtile_host::core_invocations
                  << ",\"checked_padded_elements\":" << subtile_host::checked_elements
                  << ",\"source_page_reads\":" << subtile_host::page_reads
                  << ",\"source_l1_halfword_reads\":" << subtile_host::source_l1_halfword_reads
                  << ",\"source_l1_word_reads\":" << subtile_host::source_l1_word_reads
                  << ",\"output_l1_word_writes\":" << subtile_host::output_l1_word_writes
                  << ",\"second_slot_reads\":" << subtile_host::secondary_reads << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
