/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_log.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <errno.h>

#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "spdlog/details/os-inl.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "zlib.h"

spdlog::logger* g_logger{nullptr};

std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> InitConsoleSink(const LogConfig& config)
{
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_level(static_cast<spdlog::level::level_enum>(config.consoleLevel));
    sink->set_pattern("[%l][PID:%P][TID:%t][%s][%!] %v");
    return sink;
}

std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> InitFileSink(const LogConfig& config)
{
    // after close handler
    spdlog::file_event_handlers handlers;
    handlers.after_close = [config](const std::string& filePath) {
        // rename
        static std::atomic<uint32_t> g_log_file_index{0};
        if (filePath.size() < config.fileSuffix.size() ||
            filePath.substr(filePath.size() - config.fileSuffix.size()) != config.fileSuffix) {
            HCCL_VM_ERROR("Log file name error!");
            return;
        }
        std::ostringstream oss;
        oss << filePath.substr(0, filePath.size() - config.fileSuffix.size())
            << "_" << std::to_string(g_log_file_index.fetch_add(1, std::memory_order_relaxed))
            << config.fileSuffix;
        const std::string newPath = oss.str();
        if (spdlog::details::os::rename(filePath, newPath) != 0) {
            HCCL_VM_ERROR("Fail to rename rotating log file: {} -> {} current pid {} errno: {}", filePath, newPath, getpid(), errno);
            return;
        }
    };

    std::ostringstream logFileName;
    logFileName << config.filePath << "/" << config.fileBaseName << "_" << std::to_string(getpid()) << config.fileSuffix;

    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logFileName.str(), config.maxFileSize, config.maxFiles, true, handlers);
    sink->set_level(static_cast<spdlog::level::level_enum>(config.fileLevel));
    sink->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%l][PID:%P][TID:%t][%s:%#][%!] %v");
    return sink;
}

static std::string GetLogYamlConfigPath()
{
    const char* env = std::getenv("HCCL_VM_LOG_CONFIG_PATH");
    if (env && *env) {
        return env;
    }
    return InstallPath::ResolveToInstallRoot("config/log_config.yaml");
}

LogConfig LoadLogConfig(const std::string& process_name)
{
    LogConfig cfg;

    static const std::map<std::string, std::string> yaml_node_map = {
        {"device_aarch64", "proxy"}
    };
    std::string node_name = process_name;
    auto it = yaml_node_map.find(process_name);
    if (it != yaml_node_map.end()) {
        node_name = it->second;
    }

    std::map<std::string, std::string> fields;
    if (LoadYamlStringMap(GetLogYamlConfigPath(), node_name, fields)) {
        auto it_console  = fields.find("console_level");
        auto it_file     = fields.find("file_level");
        auto it_max_size = fields.find("max_file_size");
        auto it_max_num  = fields.find("max_files");
        auto it_path     = fields.find("file_path");
        auto it_suffix   = fields.find("file_suffix");
        auto it_compress = fields.find("enable_compress");

        if (it_console  != fields.end()) {
            cfg.consoleLevel   = std::stoi(it_console->second);
        }
        if (it_file     != fields.end()) {
            cfg.fileLevel      = std::stoi(it_file->second);
        }
        if (it_max_size != fields.end()) {
            cfg.maxFileSize    = static_cast<size_t>(std::stoull(it_max_size->second));
        }
        if (it_max_num  != fields.end()) {
            cfg.maxFiles       = static_cast<size_t>(std::stoull(it_max_num->second));
        }
        if (it_path     != fields.end()) {
            cfg.filePath = InstallPath::ResolveToInstallRoot(it_path->second);
        }
        if (it_suffix   != fields.end()) {
            cfg.fileSuffix     = it_suffix->second;
        }
        if (it_compress != fields.end()) {
            cfg.enableCompress = (it_compress->second == "true" || it_compress->second == "1");
        }
    } else {
        if (process_name == "proxy") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/proxy");
        } else if (process_name == "device_aarch64") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/proxy");
        } else if (process_name == "runner") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/runner");
        } else if (process_name == "checker") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs/checker");
        } else if (process_name == "hccl_vm") {
            cfg.filePath = InstallPath::ResolveToInstallRoot("logs");
        }
    }

    if (process_name == "device_aarch64") {
        cfg.fileBaseName = "proxy_" + std::to_string(getppid());
    } else {
        cfg.fileBaseName = process_name;
    }
    return cfg;
}

void InitLogger(const LogConfig& config)
{
    try {
        auto console_sink = InitConsoleSink(config);
        auto file_sink = InitFileSink(config);
        g_logger = new spdlog::logger("muti-logger", spdlog::sinks_init_list({console_sink, file_sink}));
        g_logger->set_level(spdlog::level::trace);  // global级别设置为最低, 只通过sink的级别控制日志输出
        g_logger->flush_on(spdlog::level::warn);

        std::atexit([] (){ DeInitLogger(); });
    } catch (...) {
        HCCL_VM_ERROR("Logger init failed");
    }
}

void FlushLog()
{
    if (g_logger != nullptr) {
        g_logger->flush();
    }
}

void DeInitLogger()
{
    if (g_logger != nullptr) {
        delete g_logger;
        g_logger = nullptr;
    }
}
