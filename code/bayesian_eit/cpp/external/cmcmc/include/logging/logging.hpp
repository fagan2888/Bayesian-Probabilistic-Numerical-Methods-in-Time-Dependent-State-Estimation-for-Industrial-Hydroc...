//#include "logging/easylogging++.hpp"
#include <iostream>

#ifdef ENABLE_LOG_TRACE
	#define LOG_TRACE(x) do { \
		std::cout << x << std::endl; \
	} while (0)

#else
	#define LOG_TRACE(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_DEBUG
	#define LOG_DEBUG(x) do { \
		std::cout << x << std::endl; \
	} while (0)
#else
	#define LOG_DEBUG(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_INFO
	#define LOG_INFO(x) do { \
		std::cout << x << std::endl; \
	} while (0)
#else
	#define LOG_INFO(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_WARN
	#define LOG_WARN(x) do { \
		std::cout << x << std::endl; \
	} while (0)
#else
	#define LOG_WARN(x) do {} while (0)
#endif

#ifdef ENABLE_LOG_ERROR
	#define LOG_ERROR(x) do { \
		std:cerr << x << std::endl; \
	} while (0)
#else
	#define LOG_ERROR(x) do {} while (0)
#endif