#pragma once

#if __CUDA_ARCH__
#define IS_ON_HOST false
#define IS_ON_DEVICE true
#else
#define IS_ON_HOST true
#define IS_ON_DEVICE false
#endif