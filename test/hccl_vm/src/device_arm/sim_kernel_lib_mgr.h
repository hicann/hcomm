#ifndef SIM_KERNEL_LIB_MGR_H
#define SIM_KERNEL_LIB_MGR_H

#include <cstdint>
#include <map>
#include <string>
#include <mutex>

// 函数原型：入参void*，返回uint32_t
using KernelFn = uint32_t (*)(void*);

namespace sim {

class KernelLibManager {
public:
    static KernelLibManager& GetInstance();

    KernelFn GetOrLoadFunc(const std::string& libName, const std::string& symbolName);

    void Cleanup();

private:
    KernelLibManager();
    ~KernelLibManager();

    void* LoadKernelSo(const std::string& libName);

    KernelLibManager(const KernelLibManager&) = delete;
    KernelLibManager& operator=(const KernelLibManager&) = delete;

    void LoadBaseLibs();

    std::mutex m_mutex;
    bool m_baseLoaded{false};
    std::map<std::string, void*> m_soHandles;      // libName -> dlopen handle
    std::map<std::string, KernelFn> m_symbolCache; // "funcName" -> func ptr
};

} // namespace sim

#endif
