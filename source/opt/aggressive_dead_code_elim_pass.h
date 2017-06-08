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

#ifndef LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_
#define LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

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
class AggressiveDCEPass : public Pass {

  using cbb_ptr = const ir::BasicBlock*;

 public:
   using GetBlocksFunction =
     std::function<std::vector<ir::BasicBlock*>*(const ir::BasicBlock*)>;

  AggressiveDCEPass();
  const char* name() const override { return "aggressive-dce"; }
  Status Process(ir::Module*) override;

 private:
  // Returns the id of the merge block declared by a merge instruction in 
  // this block, if any.  If none, returns zero.
  uint32_t MergeBlockIdIfAny(const ir::BasicBlock& blk) const;

  // Compute structured successors for function |func|.
  // A block's structured successors are the blocks it branches to
  // together with its declared merge block if it has one.
  // When order matters, the merge block always appears first.
  // This assures correct depth first search in the presence of early 
  // returns and kills. If the successor vector contain duplicates
  // if the merge block, they are safely ignored by DFS.
  void ComputeStructuredSuccessors(ir::Function* func);

  // Compute structured block order |order| for function |func|. This order
  // has the property that dominators are before all blocks they dominate and
  // merge blocks are after all blocks that are in the control constructs of
  // their header.
  void ComputeStructuredOrder(
    ir::Function* func, std::list<ir::BasicBlock*>* order);

  // Return function to return ordered structure successors for a given block
  // Assumes ComputeStructuredSuccessors() has been called.
  GetBlocksFunction StructuredSuccessorsFunction();

  // If |condId| is boolean constant, return value in |condVal| and
  // |condIsConst| as true, otherwise return |condIsConst| as false.
  void GetConstCondition(uint32_t condId, bool* condVal, bool* condIsConst);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // Kill all instructions in block |bp|.
  void KillAllInsts(ir::BasicBlock* bp);

  // If block |bp| contains constant conditional branch, return true
  // and return branch and merge instructions in |branchInst| and |mergeInst|
  // and the boolean constant in |condVal|. 
  bool GetConstConditionalBranch(ir::BasicBlock* bp,
    ir::Instruction** branchInst, ir::Instruction** mergeInst,
    bool *condVal);

  // For function |func|, look for BranchConditionals with constant condition
  // and convert to a Branch to the indicated label. Delete all resulting dead
  // blocks. For all phi functions in the corresponding merge block, replace
  // all uses with id corresponding to the living predecessor. Assumes only
  // structured control flow in shader.
  bool AggressiveDCE(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Module this pass is processing
  ir::Module* module_;

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;

  // Map from block to its structured successor blocks. See 
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      block2structured_succs_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

