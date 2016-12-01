// Copyright (c) 2016 The Khronos Group Inc.
// Copyright (c) 2016 Valve Corporation
// Copyright (c) 2016 LunarG Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIBSPIRV_OPT_INLINE_PASS_H_
#define LIBSPIRV_OPT_INLINE_PASS_H_

#include <algorithm>

#include "def_use_manager.h"
#include "module.h"
#include "basic_block.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlinePass : public Pass {
 public:
  InlinePass();
  const char* name() const override { return "inline"; }
  Status Process(ir::Module*) override;

 private:
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;

  // Map from block's label id to block
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block;

  // Next unused ID
  uint32_t nextId_;

  inline void finalizeNextId(ir::Module* module) { module->SetIdBound(nextId_); }
  inline uint32_t getNextId() { return nextId_++; }

  // Exhaustively inline all function calls in func as well as in
  // all code that is inlined into func.
  bool Inline(ir::Function* func);

  // Return in newBlocks the result of inlining the call at call_ii within
  // its block call_bi. Also return in newVars additional OpVariable instructions
  // required by and to be inserted into the caller function after the block
  // call_bi is replaced with newBlocks.
  void GenInlineCode(
        std::vector<std::unique_ptr<ir::BasicBlock>>& newBlocks,
        std::vector<std::unique_ptr<ir::Instruction>>& newVars,
        ir::UptrVectorIterator<ir::Instruction> call_ii,
        ir::UptrVectorIterator<ir::BasicBlock> call_bi);

   void Initialize(ir::Module* module);
   Pass::Status ProcessImpl();

   ir::Module* module_;
   std::unique_ptr<analysis::DefUseManager> def_use_mgr_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INLINE_PASS_H_
