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
  std::unordered_map<uint32_t, ir::Function*> id2function;

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

  // Return true if indices of extract and insert match
  bool SSAMemExtInsMatch(ir::Instruction* extInst, ir::Instruction* insInst);

  // Return true if indices of extract and insert confict
  bool SSAMemExtInsConflict(ir::Instruction* extInst, ir::Instruction* insInst);

  // Looks for stores of inserts and tries to kill initial load
  bool SSAMemEliminateExtracts(ir::Function* func);

  // Looks for stores of inserts and tries to kill initial load
  bool SSAMemBreakLSCycle(ir::Function* func);

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

