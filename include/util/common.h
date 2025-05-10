#pragma once
#include <glog/logging.h>
#include <iostream>

#define div_ceil(a, b) (((a) + (b) - 1) / (b))

#define CHECK_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        LOG(ERROR) << #cmd " error, " << __FILE__ << ":" << __LINE__; \
        exit(1);\
    }\
}

enum Status {
    Succ,
    Fail,
    Timeout,
    NotFound,
    OutOfRange,
    Full
};


#define RETURN_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        LOG(ERROR) << #cmd " error, " << __FILE__ << ":" << __LINE__; \
        std::terminate();\
        return s;\
    }\
}

#define ASSERT(condition)\
     do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)

#define NotImplemented(msg) \
    do { \
        std::cerr << "Not implemented: " << msg << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::terminate(); \
    } while (false)

template <typename T>
T align_up(T value, T alignment) {
    T temp = value % alignment;
    return temp == 0? value : value - temp + alignment;
}

template <typename T>
T align_down(T value, T alignment) {
    return value - value % alignment;
}

void bind_core(int core_id);