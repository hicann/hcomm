/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SOCKET_CONFIG_H
#define HCCL_SOCKET_CONFIG_H

#include "types.h"
#include "virtual_topo.h"
#include "hash_utils.h"
#include "log.h"

namespace Hccl {
MAKE_ENUM(SocketRole, SERVER, CLIENT)

static constexpr size_t HCCP_TAG_MAX_LEN = 191; // SOCK_CONN_TAG_SIZE - 1 (留 1 字节给 '\0')
class SocketConfig {
public:
    RankId remoteRank;
    LinkData link;
    uint32_t listeningPort{DEFAULT_LISTENING_PORT};
    const std::string tag;
    uint32_t hostNic2DeviceNicMode_{0}; // 0 normal, 1: host(host cpu roce channel) - device(transport ibv)

    SocketConfig(RankId remoteRank, const LinkData& link, const std::string& tag)
        : remoteRank(remoteRank),
          link(link),
          tag(tag),
          role(link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT)
    {
        hccpTag = BuildHccpTagWithRank(tag, link, role);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          tag(tag),
          role(link.GetLocalAddr() < link.GetRemoteAddr() ? SocketRole::SERVER : SocketRole::CLIENT)
    {
        hccpTag = BuildHccpTagWithRank(tag, link, role);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag, SocketRole role, const uint32_t listenPort)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag),
          role(role)
    {
        hccpTag = BuildHccpTagByIpIndex(tag, link, role);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag, bool noRankId)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          tag(tag),
          role(link.GetLocalAddr() < link.GetRemoteAddr() ? SocketRole::SERVER : SocketRole::CLIENT),
          noRankId(noRankId)
    {
        hccpTag = BuildHccpTagByIpIndex(tag, link, role);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(
        const LinkData& link, const uint32_t listenPort, const std::string& tag, uint32_t hostNic2DeviceNicMode,
        const uint32_t myRank, const uint32_t rmtRank)
        : SocketConfig(link, listenPort, tag)
    {
        if (hostNic2DeviceNicMode == 0) {
            return;
        }
        // Parse commTag from tag prefix: tag format is "commTag_engine_X" or "commTag_engine_X_protocol_Y"
        std::string commTag = tag;
        size_t enginePos = commTag.find("_engine_");
        if (enginePos != std::string::npos) {
            commTag = commTag.substr(0, enginePos);
        } else {
            HCCL_WARNING("[SocketConfig] socketTag[%s] format error, using original tag as commTag", tag.c_str());
        }
        remoteRank = rmtRank;
        role = myRank < rmtRank ? SocketRole::SERVER : SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithRank(commTag, myRank, rmtRank, link, role);
        hostNic2DeviceNicMode_ = hostNic2DeviceNicMode;
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(const LinkData& link, const uint32_t listenPort, const std::string& tag)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag)
    {
        role = link.GetLocalAddr() < link.GetRemoteAddr() ? SocketRole::SERVER : SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithPort(tag, link, role, listenPort);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketConfig(const LinkData& link, const uint32_t listenPort, const std::string& tag, const bool isServer)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag)
    {
        role = isServer ? SocketRole::SERVER : SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithPort(tag, link, role, listenPort);
        LogIpIndexMapping(link);
        CheckAndTruncateHccpTag();
    }

    SocketRole GetRole() const { return role; }

    const string& GetHccpTag() const { return hccpTag; }

private:
    SocketRole role{};
    string hccpTag;

    // 带 rankId 的 hccpTag: tag_localRank_remoteRank_localIpIdx_remoteIpIdx (SERVER) /
    // tag_remoteRank_localRank_remoteIpIdx_localIpIdx (CLIENT)
    static string BuildHccpTagWithRank(const string& tag, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(link.GetLocalRankId()) + "_" + to_string(link.GetRemoteRankId()) + "_"
                   + to_string(link.GetLocalIpIndex()) + "_" + to_string(link.GetRemoteIpIndex());
        }
        return tag + "_" + to_string(link.GetRemoteRankId()) + "_" + to_string(link.GetLocalRankId()) + "_"
               + to_string(link.GetRemoteIpIndex()) + "_" + to_string(link.GetLocalIpIndex());
    }

    // 带显式 rankId 的 hccpTag (构造函数5: hostNic2DeviceNicMode)
    static string
    BuildHccpTagWithRank(const string& tag, uint32_t myRank, uint32_t rmtRank, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(myRank) + "_" + to_string(rmtRank) + "_" + to_string(link.GetLocalIpIndex())
                   + "_" + to_string(link.GetRemoteIpIndex());
        }
        return tag + "_" + to_string(rmtRank) + "_" + to_string(myRank) + "_" + to_string(link.GetRemoteIpIndex()) + "_"
               + to_string(link.GetLocalIpIndex());
    }

    // 仅 IP 索引的 hccpTag: tag_localIpIdx_remoteIpIdx (SERVER) / tag_remoteIpIdx_localIpIdx (CLIENT)
    static string BuildHccpTagByIpIndex(const string& tag, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(link.GetLocalIpIndex()) + "_" + to_string(link.GetRemoteIpIndex());
        }
        return tag + "_" + to_string(link.GetRemoteIpIndex()) + "_" + to_string(link.GetLocalIpIndex());
    }

    // 含 port 的 hccpTag: tag_localIpIdx_remoteIpIdx_port (SERVER) / tag_remoteIpIdx_localIpIdx_port (CLIENT)
    static string BuildHccpTagWithPort(const string& tag, const LinkData& link, SocketRole role, uint32_t listenPort)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(link.GetLocalIpIndex()) + "_" + to_string(link.GetRemoteIpIndex()) + "_"
                   + to_string(listenPort);
        }
        return tag + "_" + to_string(link.GetRemoteIpIndex()) + "_" + to_string(link.GetLocalIpIndex()) + "_"
               + to_string(listenPort);
    }

    // 打印 ipIndex 与原 IP 的映射关系
    static void LogIpIndexMapping(const LinkData& link)
    {
        HCCL_INFO(
            "[SocketConfig] hccpTag uses ipIndex[%u](localIp[%s]) ipIndex[%u](remoteIp[%s])", link.GetLocalIpIndex(),
            link.GetLocalAddr().GetIpStr().c_str(), link.GetRemoteIpIndex(), link.GetRemoteAddr().GetIpStr().c_str());
    }

    // 校验并截断 hccpTag
    void CheckAndTruncateHccpTag()
    {
        if (hccpTag.size() > HCCP_TAG_MAX_LEN) {
            HCCL_WARNING(
                "[SocketConfig] hccpTag length[%zu] exceeds max[%zu], tag[%s]", hccpTag.size(), HCCP_TAG_MAX_LEN,
                hccpTag.c_str());
            hccpTag.resize(HCCP_TAG_MAX_LEN);
        }
    }

public:
    bool noRankId{false};
};
} // namespace Hccl

namespace std {
// 特化SocketConfig的hash和equal模板，使其可用做map的key
template <>
class hash<Hccl::SocketConfig> {
public:
    size_t operator()(const Hccl::SocketConfig& socketConfig) const
    {
        auto remoteRankHash = hash<Hccl::RankId>{}(socketConfig.remoteRank);
        auto localPortHash = hash<Hccl::PortData>{}(socketConfig.link.GetLocalPort());
        auto remotePortHash = hash<Hccl::PortData>{}(socketConfig.link.GetRemotePort());
        auto tagHash = hash<string>{}(socketConfig.tag);
        auto portHash = hash<uint32_t>{}(socketConfig.listeningPort);

        return Hccl::HashCombine({remoteRankHash, localPortHash, remotePortHash, tagHash, portHash});
    }
};

template <>
class equal_to<Hccl::SocketConfig> {
public:
    bool operator()(const Hccl::SocketConfig& config, const Hccl::SocketConfig& otherConfig) const
    {
        bool IsOthersSame = config.link.GetLocalPort().GetAddr() == otherConfig.link.GetLocalPort().GetAddr()
                            && config.link.GetRemotePort().GetAddr() == otherConfig.link.GetRemotePort().GetAddr()
                            && config.tag == otherConfig.tag && config.GetHccpTag() == otherConfig.GetHccpTag()
                            && config.listeningPort == otherConfig.listeningPort;

        if (config.noRankId && otherConfig.noRankId) {
            return IsOthersSame;
        }

        return IsOthersSame && config.remoteRank == otherConfig.remoteRank;
    }
};
} // namespace std

#endif // HCCL_SOCKET_CONFIG_H
