#include "uniqueid.h"

#include <atomic>

namespace bcuda {
    std::uint64_t GetUniqueIdForThisAppInstance() {
        static std::atomic_uint64_t idCounter(0);
        return idCounter.fetch_add(1);
    }
}