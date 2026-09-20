#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace subtile_host {

constexpr uint32_t element_bytes = TEST_ELEMENT_BYTES;
constexpr uint32_t tile_bytes = 1024 * element_bytes;
constexpr uint32_t tile_words = tile_bytes / sizeof(uint32_t);
constexpr uint32_t guard_words = 16;
constexpr uint32_t guard_value = 0xBAADF00D;

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Buffer {
    uint32_t depth;
    std::vector<uint32_t> storage;
    uint32_t write_page = 0;
    uint32_t read_page = 0;
    uint32_t available = 0;
    uint32_t reserved = 0;

    explicit Buffer(uint32_t pages) : depth(pages), storage(pages * tile_words + 2 * guard_words, guard_value) {}

    uintptr_t data() const { return reinterpret_cast<uintptr_t>(storage.data() + guard_words); }

    bool contains(uintptr_t address, uint32_t bytes) const {
        return address >= data() && address + bytes <= data() + depth * tile_bytes;
    }

    void check_guards() const {
        for (uint32_t index = 0; index < guard_words; ++index) {
            require(storage[index] == guard_value, "CB prefix overwritten");
            require(storage[storage.size() - index - 1] == guard_value, "CB suffix overwritten");
        }
    }
};

struct PendingRead {
    uintptr_t destination;
    const uint8_t* source;
    uint32_t bytes;
};

struct Context {
    std::array<uint32_t, 7> arguments;
    const std::vector<uint32_t>& source;
    std::vector<uint32_t>& output;
    std::array<Buffer, 2> buffers{Buffer(8), Buffer(1)};
    std::vector<PendingRead> pending_reads;
    uint32_t writer_remaining;
    uint32_t writer_pending = 0;
    uint32_t written_pages = 0;
    uint32_t source_reads = 0;
    uint32_t second_slot_reads = 0;
    uint64_t source_l1_halfword_reads = 0;
    uint64_t source_l1_word_reads = 0;
    uint64_t output_l1_word_writes = 0;

    Context(
        std::array<uint32_t, 7> runtime_arguments,
        const std::vector<uint32_t>& source_storage,
        std::vector<uint32_t>& output_storage) :
        arguments(runtime_arguments), source(source_storage), output(output_storage), writer_remaining(arguments[1]) {}

    void release_pending() {
        auto& buffer = buffers[0];
        require(buffer.available >= writer_pending, "writer releases unavailable pages");
        buffer.available -= writer_pending;
        buffer.read_page = (buffer.read_page + writer_pending) % buffer.depth;
        writer_pending = 0;
    }

    void advance_writer() {
        auto& buffer = buffers[0];
        const uint32_t batch = std::min(4U, writer_remaining);
        if (batch == 0 || buffer.available < writer_pending + batch) {
            return;
        }
        release_pending();
        require(buffer.read_page + batch <= buffer.depth, "writer batch crosses CB ring boundary");
        const auto* first_word = reinterpret_cast<const uint32_t*>(buffer.data()) + buffer.read_page * tile_words;
        const uint64_t output_offset = static_cast<uint64_t>(arguments[2] + written_pages) * tile_words;
        require(output_offset + batch * tile_words <= output.size(), "writer output exceeds allocation");
        std::copy_n(first_word, batch * tile_words, output.data() + output_offset);
        writer_pending = batch;
        written_pages += batch;
        writer_remaining -= batch;
        if (writer_remaining == 0) {
            release_pending();
        }
    }

    void check_complete() const {
        require(pending_reads.empty(), "NOC read was not fenced");
        require(writer_remaining == 0 && writer_pending == 0, "writer pipeline did not drain");
        require(written_pages == arguments[1], "wrong output page count");
        require(source_reads <= arguments[1], "more than one source read per output page");
        require(second_slot_reads == 0, "reader accessed a second scratch page");
        require(
            output_l1_word_writes == static_cast<uint64_t>(arguments[1]) * tile_words,
            "output word written more or less than once");
        if constexpr (TEST_REPEAT_HEIGHT || (16 % TEST_REPEATS == 0 && TEST_REPEATS <= 8)) {
            require(source_l1_halfword_reads == 0, "packed row path used scalar halfword loads");
            require(source_l1_word_reads <= output_l1_word_writes, "packed row path reread output words");
            if constexpr (!TEST_REPEAT_HEIGHT) {
                require(
                    source_l1_word_reads * TEST_REPEATS <= output_l1_word_writes,
                    "aligned width path did not reuse loaded words");
            }
        }
        require(buffers[0].available == 0 && buffers[0].reserved == 0, "output CB did not drain");
        buffers[0].check_guards();
        buffers[1].check_guards();
    }
};

inline Context* active_context = nullptr;

}

constexpr uint32_t get_named_compile_time_arg_val(std::string_view name) {
    if (name == "cb_id") {
        return 0;
    }
    if (name == "scratch_cb_id") {
        return 1;
    }
    if (name == "repeat_height") {
        return TEST_REPEAT_HEIGHT;
    }
    if (name == "repeats") {
        return TEST_REPEATS;
    }
    if (name == "element_bytes") {
        return TEST_ELEMENT_BYTES;
    }
    throw "unexpected compile-time argument";
}

template <typename Value>
Value get_arg_val(uint32_t index) {
    return static_cast<Value>(subtile_host::active_context->arguments.at(index));
}

template <uint32_t Offset>
struct TensorAccessorArgs {
    constexpr uint32_t get_aligned_page_size() const { return subtile_host::tile_bytes; }
};

struct TensorAccessor {
    uint32_t address;
    uint32_t page_size;

    template <uint32_t Offset>
    TensorAccessor(TensorAccessorArgs<Offset>, uint32_t source_address, uint32_t stride) :
        address(source_address), page_size(stride) {}
};

class CircularBuffer {
public:
    explicit CircularBuffer(uint32_t identifier) : identifier_(identifier) {}

    uint32_t get_cb_id() const { return identifier_; }

    void reserve_back(uint32_t pages) {
        auto& buffer = subtile_host::active_context->buffers.at(identifier_);
        subtile_host::require(buffer.reserved == 0, "CB reserved twice without push");
        subtile_host::require(pages > 0 && buffer.available + pages <= buffer.depth, "CB would deadlock");
        subtile_host::require(buffer.write_page + pages <= buffer.depth, "CB reservation crosses ring boundary");
        buffer.reserved = pages;
    }

    uintptr_t get_write_ptr() const {
        auto& buffer = subtile_host::active_context->buffers.at(identifier_);
        subtile_host::require(buffer.reserved != 0, "write pointer obtained without reservation");
        return buffer.data() + buffer.write_page * subtile_host::tile_bytes;
    }

    void push_back(uint32_t pages) {
        auto& context = *subtile_host::active_context;
        auto& buffer = context.buffers.at(identifier_);
        subtile_host::require(identifier_ == 0, "scratch CB must stay local to reader");
        subtile_host::require(buffer.reserved == pages, "CB push does not match reservation");
        subtile_host::require(context.pending_reads.empty(), "CB published before NOC barrier");
        buffer.reserved = 0;
        buffer.available += pages;
        buffer.write_page = (buffer.write_page + pages) % buffer.depth;
        context.advance_writer();
    }

private:
    uint32_t identifier_;
};

class Noc {
public:
    struct SourceArgs {
        uint32_t page_id;
        uint32_t offset_bytes = 0;
    };
    struct DestinationArgs {
        uint32_t offset_bytes;
    };

    void async_read(
        const TensorAccessor& source,
        const CircularBuffer& destination,
        uint32_t bytes,
        SourceArgs source_args,
        DestinationArgs destination_args) {
        auto& context = *subtile_host::active_context;
        const auto& buffer = context.buffers.at(destination.get_cb_id());
        subtile_host::require(source.address == context.arguments[0], "stale source address");
        const uint64_t source_offset =
            static_cast<uint64_t>(source_args.page_id) * source.page_size + source_args.offset_bytes;
        subtile_host::require(
            source_offset + bytes <= context.source.size() * sizeof(uint32_t), "NOC source out of bounds");
        const uintptr_t destination_address = destination.get_write_ptr() + destination_args.offset_bytes;
        subtile_host::require(buffer.contains(destination_address, bytes), "NOC destination out of bounds");
        subtile_host::require(
            destination_args.offset_bytes + bytes <= buffer.reserved * subtile_host::tile_bytes,
            "NOC read exceeds reservation");
        context.pending_reads.push_back(
            {destination_address, reinterpret_cast<const uint8_t*>(context.source.data()) + source_offset, bytes});
        ++context.source_reads;
        context.second_slot_reads += destination_args.offset_bytes != 0;
    }

    void async_read_barrier() {
        auto& context = *subtile_host::active_context;
        for (const auto& read : context.pending_reads) {
            std::memcpy(reinterpret_cast<void*>(read.destination), read.source, read.bytes);
        }
        context.pending_reads.clear();
    }
};

template <typename Value>
class CoreLocalMem {
public:
    explicit CoreLocalMem(uintptr_t address) : address_(address) {}

    Value& operator[](uint32_t index) const {
        auto& context = *subtile_host::active_context;
        subtile_host::require(context.pending_reads.empty(), "L1 accessed before NOC read barrier");
        const uintptr_t address = address_ + static_cast<uintptr_t>(index) * sizeof(Value);
        if (context.buffers[1].contains(address, sizeof(Value))) {
            if constexpr (sizeof(Value) == 2) {
                ++context.source_l1_halfword_reads;
            } else {
                static_assert(sizeof(Value) == 4);
                ++context.source_l1_word_reads;
            }
        } else {
            const auto& output = context.buffers[0];
            const uintptr_t write_start = output.data() + output.write_page * subtile_host::tile_bytes;
            subtile_host::require(
                sizeof(Value) == 4 && output.contains(address, sizeof(Value)) && address >= write_start &&
                    address + sizeof(Value) <= write_start + output.reserved * subtile_host::tile_bytes,
                "core-local output access exceeds the current reservation");
            ++context.output_l1_word_writes;
        }
        return *reinterpret_cast<Value*>(address);
    }

private:
    uintptr_t address_;
};
