// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_
#define LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_


#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class DeadBranchElimPass : public Pass {
 public:
  DeadBranchElimPass();
  const char* name() const override { return "sroa"; }
  Status Process(ir::Module*) override;

 private:
  // If |condId| is boolean constant, return value in |condVal| and
  // |condIsConst| as true, otherwise return |condIsConst| as false.
  void GetConstCondition(uint32_t condId, bool* condVal, bool* condIsConst);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // Kill all instructions in block |bp|.
  void KillBlk(ir::BasicBlock* bp);

  // Return merge block label id if |block_ptr| is loop header. Otherwise
  // return 0.
  uint32_t GetMergeBlkId(ir::BasicBlock* block_ptr);

  // For function |func|, look for BranchConditionals with constant condition
  // and convert to a Branch to the indicted label. The BranchConditional must
  // be preceeded by OpSelectionMerge. For all phi functions in merge block,
  // replace all uses with id corresponding to living predecessor.
  bool EliminateDeadBranches(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Module this pass is processing
  ir::Module* module_;

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_

