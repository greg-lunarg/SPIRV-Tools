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
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Map from block's label id to block
  std::unordered_map<uint32_t, ir::BasicBlock*> label2block_;

  // Map from block's label id to its predecessor blocks ids
  std::unordered_map<uint32_t, std::vector<uint32_t>> label2preds_;

  // Map from block's label id to its SSA map
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> label2SSA_;

  // Map from SSA Variable to its single store
  std::unordered_map<uint32_t, ir::Instruction*> ssaVars;

  // Hash for CompKey
  struct pairhash {
    public:
      template <typename T, typename U>
      std::size_t operator()(const std::pair<T, U> &x) const
      {
          return (std::hash<T>()(x.first) << 8) ^ std::hash<U>()(x.second);
      }
  };

  // Map from SSA Component (Var, Index pair) to its single store
  typedef std::pair<uint32_t, uint32_t> CompKey;
  std::unordered_map<CompKey, ir::Instruction*, pairhash> ssaComps;

  // Map from store to its instruction index
  std::unordered_map<ir::Instruction*, uint32_t> storeIdx;

  // Set of SSA Component Variables
  std::unordered_set<uint32_t> ssaCompVars;

  // Set of non-SSA Variables
  std::unordered_set<uint32_t> nonSsaVars;

  // Set of verified target types
  std::unordered_set<uint32_t> seenTargetVars;

  // Set of verified non-target types
  std::unordered_set<uint32_t> seenNonTargetVars;

  // Set of function scope variables for current function
  std::unordered_set<uint32_t> funcVars;

  // Set of label ids of visited blocks
  std::unordered_set<uint32_t> visitedBlocks;

  // Map from variable to its live store in block
  std::unordered_map<uint32_t, ir::Instruction*> sbVarStores;

  // Map from variable to its live load in block
  std::unordered_map<uint32_t, ir::Instruction*> sbVarLoads;

  // Map from component (var, index pair) to its live store in block
  std::unordered_map<CompKey, ir::Instruction*, pairhash> sbCompStores;

  // Set of undeletable variables
  std::unordered_set<uint32_t> sbPinnedVars;

  // Map from component (var, index pair) to its live store in block
  std::unordered_set<CompKey, pairhash> sbPinnedComps;

  // Returns true if type is a scalar type
  // or a vector or matrix
  bool isMathType(const ir::Instruction* typeInst);

  // Returns true if type is a scalar, vector, matrix
  // or struct of only those types
  bool isTargetType(const ir::Instruction* typeInst);

  // Next unused ID
  uint32_t nextId_;

  inline void finalizeNextId(ir::Module* module) {
    module->SetIdBound(nextId_);
  }

  inline uint32_t getNextId() {
    return nextId_++;
  }

  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t& varId);

  bool isTargetVar(uint32_t varId);

  // Find all function scope variables that are stored to only once
  // and create two maps: one for full variable stores and one for
  // component stores. They will map variable (and component index)
  // to the store value Id. Also cache all variables that
  // are not SSA. Only analyze variables of scalar, vector, 
  // matrix types and struct types containing only these types.
  void SSAMemAnalyze(ir::Function* func);

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
  bool SSAMemProcess(ir::Function* func);

  // Return true if varId is not a function variable or if it has
  // a load
  bool isLiveVar(uint32_t varId);

  // Return true if Store is not to function variable or if its
  // base variable has a load
  bool isLiveStore(ir::Instruction* storeInst);

  // Delete inst and iterate DCE on all its operands 
  void DCEInst(ir::Instruction* inst);

  // Remove all stores to useless SSA variables. Remove useless
  // access chains and variables as well. Assumes Analyze and
  // Process has been run.
  bool SSAMemDCE();

  void SBEraseComps(uint32_t varId);

  // Do single block memory optimization of function variables.
  // For loads, if previous load or store to same component,
  // replace load id with previous id and delete load. Finally,
  // check if remaining stores are useless, and delete store
  // and variable.
  bool SSAMemSingleBlock(ir::Function* func);

  // Perform DCEInst on all instructions in function
  bool SSAMemDCEFunc(ir::Function* func);

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

  // Convert all access chain loads and stores into extracts and
  // inserts.
  bool SSAMemAccessChainRemoval(ir::Function* func);

  // Return true if function control flow is structured
  bool IsStructured(ir::Function* func);

  // Return true if loop header block
  bool IsLoopHeader(ir::BasicBlock* block_ptr);

  // Copy SSA map from predecessor. No phis generated.
  void SSABlockInitSinglePred(ir::BasicBlock* block_ptr);

  // Return true if variable is stored in the label range
  bool HasStore(uint32_t var_id, uint32_t first_label, uint32_t last_label);

  // Return true if variable is loaded after the label
  bool HasLoad(uint32_t var_id, uint32_t label);

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

  // Remove remaining loads and stores of targeted function scope variables
  // in func. Insert Phi functions where necessary. Assumes that AccessChainRemoval
  // and SingleBlock have already been run. Running SingleStore beforehand will
  // make this more efficient.
  bool SSAMemSSARewrite(ir::Function* func);

  // Return true if indices of extract and insert match
  bool SSAMemExtInsMatch(ir::Instruction* extInst, ir::Instruction* insInst);

  // Return true if indices of extract and insert confict
  bool SSAMemExtInsConflict(ir::Instruction* extInst, ir::Instruction* insInst);

  // Looks for stores of inserts and tries to kill initial load
  bool SSAMemEliminateExtracts(ir::Function* func);

  // Look for cycles of load/inserts/store where there is only the single
  // load and store of that (function scope) variable. If all inserts except
  // the last have one use, delete the store and change the load to an undef.
  // AccessChainRemoval creates these cycles and they must be specially
  // detected and broken to allow DCE to happen.
  bool SSAMemBreakLSCycle(ir::Function* func);
  
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
  bool SSAMemDeadBranchEliminate(ir::Function* func);

  // If a block branches to another block and no other block branches
  // to that block, the two blocks can be merged. This is primarily
  // cleanup after dead branch elimination. Merging blocks can also
  // create additional memory optimization opportunities.
  bool SSAMemBlockMerge(ir::Function* func);

  // For each load of SSA variable, replace all uses of the load
  // with the value stored, if possible. Return true if the any
  // instructions are modified. 
  bool SSAMem(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  ir::Module* module_;
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSAMEM_PASS_H_

