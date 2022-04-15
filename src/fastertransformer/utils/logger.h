#pragma once

#include <cstdlib>
#include <map>
#include <string>

#include "src/fastertransformer/utils/string_utils.h"

namespace fastertransformer {

class Logger {

public:
    enum Level {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger& getLogger()
    {
        static Logger instance;
        return instance;
    }
    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    template<typename... Args>
    void log(const Level level, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt = getPrefix(level) + format + "\n";
            FILE* out = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    template<typename... Args>
    void log(const Level level, const int rank, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt = getPrefix(level, rank) + format + "\n";
            FILE* out = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    void setLevel(const Level level)
    {
        level_ = level;
        log(INFO, "Set logger level by %s", getLevelName(level).c_str());
    }

private:
    const std::string PREFIX = "[FT]";
    std::map<Level, std::string> level_name_ = {
        {TRACE, "TRACE"}, {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"}};

#ifndef NDEBUG
    const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
    const Level DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;

    Logger()
    {
        char* level_name = std::getenv("FT_LOG_LEVEL");
        if (level_name != nullptr) {
            std::map<std::string, Level> name_to_level = {
                {"TRACE", TRACE},
                {"DEBUG", DEBUG},
                {"INFO", INFO},
                {"WARNING", WARNING},
                {"ERROR", ERROR},
            };
            auto level = name_to_level.find(level_name);
            if (level != name_to_level.end()) {
                setLevel(level->second);
            }
            else {
                fprintf(stderr,
                        "[FT][WARNING] Invalid logger level FT_LOG_LEVEL=%s. "
                        "Ignore the environment variable and use a default "
                        "logging level.\n",
                        level_name);
                level_name = nullptr;
            }
        }
    }

    inline std::string getLevelName(const Level level)
    {
        return level_name_[level];
    }

    inline std::string getPrefix(const Level level)
    {
        return PREFIX + "[" + getLevelName(level) + "] ";
    }

    inline std::string getPrefix(const Level level, const int rank)
    {
        return PREFIX + "[" + getLevelName(level) + "][" + std::to_string(rank) + "] ";
    }
};

#define FT_LOG(level, ...) fastertransformer::Logger::getLogger().log(level, __VA_ARGS__)
#define FT_LOG_TRACE(...) FT_LOG(fastertransformer::Logger::TRACE, __VA_ARGS__)
#define FT_LOG_DEBUG(...) FT_LOG(fastertransformer::Logger::DEBUG, __VA_ARGS__)
#define FT_LOG_INFO(...) FT_LOG(fastertransformer::Logger::INFO, __VA_ARGS__)
#define FT_LOG_WARNING(...) FT_LOG(fastertransformer::Logger::WARNING, __VA_ARGS__)
#define FT_LOG_ERROR(...) FT_LOG(fastertransformer::Logger::ERROR, __VA_ARGS__)
}  // namespace fastertransformer
