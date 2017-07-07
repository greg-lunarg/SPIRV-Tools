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

#ifndef LIBSPIRV_OPT_COMMON_UNIFORM_ELIM_PASS_H_
#define LIBSPIRV_OPT_COMMON_UNIFORM_ELIM_PASS_H_

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <utility>
#include <queue>

#include "def_use_manager.h"
#include "module.h"
#include "basic_block.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CommonUniformElimPass : public Pass {
 public:
  CommonUniformElimPass();
  const char* name() const override { return "common-uniform-elim"; }
  Status Process(ir::Module*) override;

 private:
  // Returns true if |opcode| is a non-ptr access chain op
  bool IsNonPtrAccessChain(const SpvOp opcode) const;

  // Return true if |block_ptr| is loop header block
  bool IsLoopHeader(ir::BasicBlock* block_ptr);

  // Returns the id of the merge block declared by a merge instruction in 
  // block |blk|, if any.  If none, returns zero.
  uint32_t MergeBlockIdIfAny(const ir::BasicBlock& blk) const;

  // Given a load or store pointed at by |ip|, return the pointer
  // instruction. Also return the variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if variable is uniform
  bool IsUniformVar(uint32_t varId);

  // Delete inst if it has no uses. Assumes inst has a resultId.
  void DeleteIfUseless(ir::Instruction* inst);

  // Replace all instances of load's id with replId and delete load
  // and its access chain, if any
  void ReplaceAndDeleteLoad(ir::Instruction* loadInst,
    uint32_t replId,
    ir::Instruction* ptrInst);

  // Return type id for pointer's pointee
  uint32_t GetPointeeTypeId(const ir::Instruction* ptrInst);

  // For the (constant index) access chain ptrInst, create an
  // equivalent load and extract
  void GenACLoadRepl(const ir::Instruction* ptrInst,
      std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
      uint32_t& resultId);

  // Return true if all indices are constant
  bool IsConstantIndexAccessChain(ir::Instruction* acp);

  // Convert all uniform access chain loads into load/extract.
  bool UniformAccessChainConvert(ir::Function* func);

  // Eliminate loads of uniform variables which have previously been loaded.
  // If first load is in control flow, move it to first block of function.
  // Most effective if preceded by UniformAccessChainRemoval().
  bool CommonUniformLoadElimination(ir::Function* func);

  // Eliminate duplicated extracts of same id. Extract may be moved to same
  // block as the id definition. This is primarily intended for extracts
  // from uniform loads. Most effective if preceded by
  // CommonUniformLoadElimination().
  bool CommonExtractElimination(ir::Function* func);

  // Initialize the label2SSA map entry for a block. Insert phi instructions
  // into block when necessary. All predecessor blocks must have been
  // For function |func|, first change all constant index
  // access chain loads into equivalent composite extracts. Then consolidate 
  // identical uniform loads into one uniform load. Finally, consolidate
  // identical uniform extracts into one uniform extract. This may require
  // moving a load or extract to a point which dominates all uses.
  // Return true if func is modified. 
  //
  // This pass requires the function to have structured control flow ie shader
  // capability. It also requires logical addressing ie Addresses capability
  // is not enabled. It also currently does not support any extensions.
  bool EliminateCommonUniform(ir::Function* func);

  // Return true if all extensions in this module are allowed by this pass.
  // Currently, no extensions are supported.
  // TODO(greg-lunarg): Add extensions to supported list.
  bool AllExtensionsSupported() const;

  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Module this pass is processing
  ir::Module* module_;

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  std::unordered_map<uint32_t, uint32_t> uniform2load_id_;

  // Map of extract composite ids to map of indices to insts
  std::unordered_map<uint32_t, std::unordered_map<uint32_t,
      std::list<ir::Instruction*>>> comp2idx2inst_;

  // Next unused ID
  uint32_t next_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSAMEM_PASS_H_

