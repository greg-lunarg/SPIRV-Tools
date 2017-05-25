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

#ifndef LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_
#define LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_


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
class LocalSingleStoreElimPass : public Pass {
  using cbb_ptr = const ir::BasicBlock*;

 public:
  LocalSingleStoreElimPass();
  const char* name() const override { return "sroa"; }
  Status Process(ir::Module*) override;

 private:
   // Module this pass is processing
   ir::Module* module_;

   // Def-Uses for the module we are processing
   std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Map from block's label id to block
  std::unordered_map<uint32_t, ir::BasicBlock*> label2block_;

  // Map from SSA Variable to its single store
  std::unordered_map<uint32_t, ir::Instruction*> ssa_var2store_;

  // Map from store to its ordinal position in its block.
  std::unordered_map<ir::Instruction*, uint32_t> store2idx_;

  // Map from store to its block.
  std::unordered_map<ir::Instruction*, ir::BasicBlock*> store2blk_;

  // Set of non-SSA Variables
  std::unordered_set<uint32_t> non_ssa_vars_;

  // Set of verified target types
  std::unordered_set<uint32_t> seen_target_vars_;

  // Set of verified non-target types
  std::unordered_set<uint32_t> seen_non_target_vars_;

  // Augmented CFG Entry Block
  ir::BasicBlock pseudo_entry_block_;

  // Augmented CFG Exit Block
  ir::BasicBlock pseudo_exit_block_;

  // CFG Predecessors
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
    predecessors_map_;

  // CFG Successors
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
    successors_map_;

  // CFG Augmented Predecessors
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
    augmented_predecessors_map_;

  // CFG Augmented Successors
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
    augmented_successors_map_;

  // Immediate Dominator Map
  // If block has no idom it points to itself.
  std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> idom_;

  // Next unused ID
  uint32_t next_id_;

  // Save next available id into |module|.
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Return next available id and generate next.
  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  // Returns true if |typeInst| is a scalar type
  // or a vector or matrix
  bool IsMathType(const ir::Instruction* typeInst);

  // Returns true if |typeInst| is a math type or a struct or array
  // of a math type.
  bool IsTargetType(const ir::Instruction* typeInst);

  // Given a load or store, return the pointer instruction.
  // Also return the variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if |varId| is function scope variable of targeted type.
  bool IsTargetVar(uint32_t varId);

  // Find all function scope variables in |func| that are stored to
  // only once (SSA) and map to their stored value id. Only analyze
  // variables of scalar, vector, matrix types and struct and array
  // types comprising only these types. Currently this analysis is
  // is not done in the presence of function calls. TODO(): Allow
  // analysis in the prensence of function calls.
  void SingleStoreAnalyze(ir::Function* func);

  // Delete |inst| if it has no uses. Assumes |inst| has a non-zero resultId.
  void DeleteIfUseless(ir::Instruction* inst);

  // Replace all instances of |loadInst|'s id with |replId| and delete
  // |loadInst| and its pointer |ptrInst| if it is a useless access chain.
  void ReplaceAndDeleteLoad(ir::Instruction* loadInst,
    uint32_t replId,
    ir::Instruction* ptrInst);

  using GetBlocksFunction =
    std::function<const std::vector<ir::BasicBlock*>*(const ir::BasicBlock*)>;

  /// Returns the block successors function for the augmented CFG.
  GetBlocksFunction AugmentedCFGSuccessorsFunction() const;

  /// Returns the block predecessors function for the augmented CFG.
  GetBlocksFunction AugmentedCFGPredecessorsFunction() const;

  // Calculate immediate dominators for |func|'s CFG. Leaves result
  // in idom_. Entries for augmented CFG (pseudo blocks) are not created.
  void CalculateImmediateDominators(ir::Function* func);
  
  // Return true if instruction in |blk0| at ordinal position |idx0|
  // dominates instruction in |blk1| at position |idx1|.
  bool Dominates(ir::BasicBlock* blk0, uint32_t idx0,
    ir::BasicBlock* blk1, uint32_t idx1);

  // For each load of an SSA variable in |func|, replace all uses of
  // the load with the value stored if the store dominates the load.
  // Assumes that SingleStoreAnalyze() has just been run. Return true
  // if any instructions are modified.
  bool SingleStoreProcess(ir::Function* func);

  // Return true if any instruction loads from |varId|
  bool HasLoads(uint32_t varId);

  // Return true if |varId| is not a function variable or if it has
  // a load
  bool IsLiveVar(uint32_t varId);

  // Return true if |storeInst| is not a function variable or if its
  // base variable has a load
  bool IsLiveStore(ir::Instruction* storeInst);

  // Delete |inst| and iterate DCE on all its operands if they are now
  // useless.
  void DCEInst(ir::Instruction* inst);

  // Remove all stores to useless SSA variables. Remove useless
  // access chains and variables as well. Assumes SingleStoreAnalyze
  // and SingleStoreProcess has been run.
  bool SingleStoreDCE();

  // Do "single-store" optimization of function variables defined only
  // with a single non-access-chain store in |func|. Replace all their
  // non-access-chain loads with the value that is stored and eliminate
  // any resulting dead code.
  bool LocalSingleStoreElim(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_

