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

#ifndef LIBSPIRV_OPT_SSAMEM_PASS_H_
#define LIBSPIRV_OPT_SSAMEM_PASS_H_


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
class SSAMemPass : public Pass {
 public:
  SSAMemPass();
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

  // Map from block's label id to its predecessor blocks ids
  std::unordered_map<uint32_t, std::vector<uint32_t>> label2preds_;

  // Map from block's label id to its SSA map
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
      label2ssa_map_;

  // Map from SSA Variable to its single store
  std::unordered_map<uint32_t, ir::Instruction*> ssa_var2store_;

  // Map from store to its ordinal position in the function.
  std::unordered_map<ir::Instruction*, uint32_t> store2idx_;

  // Set of non-SSA Variables
  std::unordered_set<uint32_t> non_ssa_vars_;

  // Set of verified target types
  std::unordered_set<uint32_t> seen_target_vars_;

  // Set of verified non-target types
  std::unordered_set<uint32_t> seen_non_target_vars_;

  // Map from type to undef for current function
  std::unordered_map<uint32_t, uint32_t> type2undefs_;

  // Set of label ids of visited blocks
  std::unordered_set<uint32_t> visitedBlocks;

  // Map from variable to its live store in block
  std::unordered_map<uint32_t, ir::Instruction*> var2store_;

  // Map from variable to its live load in block
  std::unordered_map<uint32_t, ir::Instruction*> var2load_;

  // Set of undeletable variables
  std::unordered_set<uint32_t> pinned_vars_;

  // Map from block id to loop. 0 indicates no loop.
  std::unordered_map<uint32_t, uint32_t> block2loop_;

  // Map from loop to last block.
  std::unordered_map<uint32_t, uint32_t> loop2last_block_;

  // Map from block to ordinal
  std::unordered_map<uint32_t, uint32_t> block2ord_;

  // Map from variable to last load block
  std::unordered_map<uint32_t, uint32_t> var2last_load_block_;

  // Map from variable to last live block
  std::unordered_map<uint32_t, uint32_t> var2last_live_block_;

  // Map from uniform to load id
  std::unordered_map<uint32_t, uint32_t> uniform2load_id_;

  // Map of extract composite ids to map of indices to insts
  std::unordered_map<uint32_t, std::unordered_map<uint32_t,
      std::list<ir::Instruction*>>> comp2idx2inst_;

  // Set of live function ids
  std::set<uint32_t> live_func_ids_;

  // Stack of called function ids
  std::queue<uint32_t> called_func_ids_;

  // Next unused ID
  uint32_t next_id_;

  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  // Returns true if type is a scalar type
  // or a vector or matrix
  bool IsMathType(const ir::Instruction* typeInst);

  // Returns true if type is a scalar, vector, matrix
  // or struct of only those types
  bool IsTargetType(const ir::Instruction* typeInst);

  // Given a load or store pointed at by |ip|, return the pointer
  // instruction. Also return the variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t& varId);

  // Return true if variable is uniform
  bool IsUniformVar(uint32_t varId);

  // Return true if variable is math type, or vector or matrix
  // of target type, or struct or array of target type
  bool IsTargetVar(uint32_t varId);

  // Find all function scope variables that are stored to only once
  // and create two maps: one for full variable stores and one for
  // component stores. They will map variable (and component index)
  // to the store value Id. Also cache all variables that
  // are not SSA. Only analyze variables of scalar, vector, 
  // matrix types and struct types containing only these types.
  void SingleStoreAnalyze(ir::Function* func);

  // Delete inst if it has no uses. Assumes inst has a resultId.
  void DeleteIfUseless(ir::Instruction* inst);

  // Replace all instances of load's id with replId and delete load
  // and its access chain, if any
  void ReplaceAndDeleteLoad(ir::Instruction* loadInst,
    uint32_t replId,
    ir::Instruction* ptrInst);

  // For each load of SSA variable, replace all uses of the load
  // with the value stored, if possible. Assumes that SSAMemAnalyze
  // has just been run for func. Return true if the any
  // instructions are modified.
  bool SingleStoreProcess(ir::Function* func);

  // Return true if |varId| has a load
  bool HasLoads(uint32_t varId);

  // Return true if |varId| is not a function variable or if it has
  // a load
  bool IsLiveVar(uint32_t varId);

  // Return true if Store is not to function variable or if its
  // base variable has a load
  bool IsLiveStore(ir::Instruction* storeInst);

  // Delete inst and iterate DCE on all its operands 
  void DCEInst(ir::Instruction* inst);

  // Remove all stores to useless SSA variables. Remove useless
  // access chains and variables as well. Assumes SingleStoreAnalyze
  // and SingleStoreProcess has been run.
  bool SingleStoreDCE();

  // Do "single-store" optimization of function variables defined only
  // with a single non-access-chain store. Replace all its non-access-
  // chain loads with the value that is stored.
  // TODO(): Add requirement that store dominates load. Until then,
  // the generated code is not incorrect, but we can lose the fact
  // that the load is ultimately undefined.
  bool LocalSingleStoreElim(ir::Function* func);

  // Do single block memory optimization of function variables
  // referenced only with non-access-chain loads and stores. For
  // loads, if previous load or store to same component, replace
  // load id with previous id and delete load. Finally, check if
  // remaining stores are useless, and delete store and variable.
  bool LocalSingleBlockElim(ir::Function* func);

  // Exhaustively remove all instructions whose result ids are not used and all
  // stores of function scope variables that are not loaded.
  bool FuncDCE(ir::Function* func);

  // Return type id for pointer's pointee
  uint32_t GetPteTypeId(const ir::Instruction* ptrInst);

  // Create a load/insert/store equivalent to a store of
  // valId through ptrInst.
  void GenACStoreRepl(const ir::Instruction* ptrInst,
      uint32_t valId,
      std::vector<std::unique_ptr<ir::Instruction>>& newInsts);

  // For the (constant index) access chain ptrInst, create an
  // equivalent load and extract
  void GenACLoadRepl(const ir::Instruction* ptrInst,
      std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
      uint32_t& resultId);

  // Return true if all indices are constant
  bool IsConstantIndexAccessChain(ir::Instruction* acp);

  // Identify all function scope variables which are accessed only
  // with loads, stores and access chains with constant indices.
  // Convert all loads and stores of such variables into equivalent
  // loads, stores, extracts and inserts. This unifies access to these
  // variables to a single mode and simplifies analysis and optimization.
  bool LocalAccessChainConvert(ir::Function* func);

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

  // Return true if function control flow is structured
  bool IsStructured(ir::Function* func);

  // Return true if loop header block
  uint32_t GetMergeBlkId(ir::BasicBlock* block_ptr);

  // Return true if loop header block
  bool IsLoopHeader(ir::BasicBlock* block_ptr);

  // Copy SSA map from predecessor. No phis generated.
  void SSABlockInitSinglePred(ir::BasicBlock* block_ptr);

  // Initialize data structures used by IsLoaded
  void InitSSARewrite(ir::Function& func);

  // Return true if variable is loaded after the label
  bool IsLiveAfter(uint32_t var_id, uint32_t label);

  void SSABlockInitLoopHeader(ir::UptrVectorIterator<ir::BasicBlock> block_itr);

  // Merge SSA Maps from all predecessors. If any variables are missing
  // in any predecessors maps, remove that variable from the resulting map.
  // If any value ids differ for any variable, create a phi function and
  // use that value id for the variable in the resulting map. Assumes all
  // predecessors have been visited by SSARewrite.
  void SSABlockInitSelectMerge(ir::BasicBlock* block_ptr);

  // Initialize the label2SSA map entry for a block. Insert phi instructions
  // into block when necessary. All predecessor blocks must have been
  // visited by SSARewrite except for backedges.
  void SSABlockInit(ir::UptrVectorIterator<ir::BasicBlock> block_itr);

  // Return undef in function for type. Create and insert an undef after the
  // first non-variable in the function if it doesn't already exist. Add
  // undef to function undef map.
  uint32_t Type2Undef(uint32_t type_id);

  // Patch phis in loop header block now that the map is complete for the
  // backedge predecessor. Specifically, for each phi, find the value
  // corresponding to the backedge predecessor. That contains the variable id
  // that this phi corresponds to. Change this phi operand to the the value
  // which corresponds to that variable in the predecessor map.
  void PatchPhis(uint32_t header_id, uint32_t back_id);

  // Remove remaining loads and stores of function scope variables only
  // referenced with non-access-chain loads and stores. Insert Phi functions
  // where necessary. Assumes that LocalAccessChainRemoval and
  // SingleBlockLocalElim have already been run. Running SingleStoreLocalElim
  // beforehand will make this more efficient.
  bool LocalSSARewrite(ir::Function* func);

  // Return true if indices of extract and insert match
  bool SSAMemExtInsMatch(ir::Instruction* extInst, ir::Instruction* insInst);

  // Return true if indices of extract and insert confict
  bool SSAMemExtInsConflict(ir::Instruction* extInst, ir::Instruction* insInst);

  // Look for OpExtract on sequence of OpInserts. If there is an insert
  // with identical indices, replace the extract with the value that is inserted.
  bool InsertExtractElim(ir::Function* func);

  // Look for cycles of load/inserts/store where there is only the single
  // load and store of that (function scope) variable. If all inserts except
  // the last have one use, delete the store and change the load to an undef.
  // AccessChainRemoval creates these cycles and breaking them allows
  // DCE to happen on them.
  bool InsertCycleBreak(ir::Function* func);
  
  // If condId is boolean constant, return value and condIsConst as true,
  // otherwise return condIsConst as false.
  void SSAMemGetConstCondition(uint32_t condId, bool* condVal, bool* condIsConst);

  // Add branch to end of block bp
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // Kill all instructions in block bp
  void SSAMemKillBlk(ir::BasicBlock* bp);

  // Look for BranchConditionals with constant condition and convert
  // to a branch. Fix phi functions in block whose branch is eliminated.
  // Eliminate preceding OpSelectionMerge if it exists.
  bool DeadBranchEliminate(ir::Function* func);

  // If a block branches to another block and no other block branches
  // to that block, the two blocks can be merged. This is primarily
  // cleanup after dead branch elimination. Merging blocks can also
  // create additional memory optimization opportunities.
  bool BlockMerge(ir::Function* func);

  // Mark called functions as live
  void FindCalledFuncs(uint32_t funcId);

  // Remove all dead functions from module. Only retain entry point
  // and exported functions and the functions they call.
  bool DeadFunctionElim();

  // For each load of SSA variable, replace all uses of the load
  // with the value stored, if possible. Return true if the any
  // instructions are modified. 
  bool SSAMem(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSAMEM_PASS_H_

