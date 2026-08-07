/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_FUNC_BLOCK_H
#define HCOMM_CCU_FUNC_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_funcblock_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

    class FuncBlock {
    public:
        FuncBlock(CcuRepContext* context, std::string label, uint16_t callLayer = FUNC_CALL_LAYER_INVALID);
        ~FuncBlock();

        template <typename... Arguments>
        FuncBlock& operator()(const Arguments&... args)
        {
            DefineInArgHelper(args...);
            return *this;
        }

        template <typename T>
        void DefineInArg(T&& arg)
        {
            repFuncBlock->DefineInArg(std::forward<T>(arg));
        }

        template <typename T>
        void DefineOutArg(T&& arg)
        {
            repFuncBlock->DefineOutArg(std::forward<T>(arg));
        }

    private:
        template <typename First>
        void DefineInArgHelper(const First& first)
        {
            repFuncBlock->DefineInArg(first);
        }

        template <typename First, typename... Rest>
        void DefineInArgHelper(const First& first, const Rest&... rest)
        {
            repFuncBlock->DefineInArg(first);
            DefineInArgHelper(rest...);
        }

        CcuRepContext* context{nullptr};
        std::string label;

        std::shared_ptr<CcuRepFuncBlock> repFuncBlock{nullptr};
        std::shared_ptr<CcuRepBlock> curActiveBlock{nullptr};

        uint16_t callLayer{FUNC_CALL_LAYER_INVALID};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_FUNC_BLOCK_H
