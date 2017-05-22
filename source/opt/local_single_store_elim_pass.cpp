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

#include "local_single_store_elim_pass.h"
#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvStorePtrId = 0;
static const int kSpvStoreValId = 1;
static const int kSpvLoadPtrId = 0;
static const int kSpvAccessChainPtrId = 0;
static const int kSpvTypePointerStorageClass = 0;
static const int kSpvTypePointerTypeId = 1;

namespace spvtools {
namespace opt {

bool LocalSingleStoreElimPass::IsMathType(const ir::Instruction* typeInst) {
  switch (typeInst->opcode()) {
  case SpvOpTypeInt:
  case SpvOpTypeFloat:
  case SpvOpTypeBool:
  case SpvOpTypeVector:
  case SpvOpTypeMatrix:
    return true;
  default:
    break;
  }
  return false;
}

bool LocalSingleStoreElimPass::IsTargetType(const ir::Instruction* typeInst) {
  if (IsMathType(typeInst))
    return true;
  if (typeInst->opcode() != SpvOpTypeStruct &&
      typeInst->opcode() != SpvOpTypeArray)
    return false;
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](const uint32_t* tid) {
    ir::Instruction* compTypeInst =
        def_use_mgr_->id_to_defs().find(*tid)->second;
    // Ignore length operand in Array type
    if (compTypeInst->opcode() == SpvOpConstant) return;
    if (!IsMathType(compTypeInst)) ++nonMathComp;
  });
  return nonMathComp == 0;
}

ir::Instruction* LocalSingleStoreElimPass::GetPtr(ir::Instruction* ip, uint32_t* varId) {
  const uint32_t ptrId = ip->opcode() == SpvOpStore ?
      ip->GetSingleWordInOperand(kSpvStorePtrId) :
      ip->GetSingleWordInOperand(kSpvLoadPtrId);
  ir::Instruction* ptrInst =
    def_use_mgr_->id_to_defs().find(ptrId)->second;
  *varId = ptrInst->opcode() == SpvOpAccessChain ?
    ptrInst->GetSingleWordInOperand(kSpvAccessChainPtrId) :
    ptrId;
  return ptrInst;
}

bool LocalSingleStoreElimPass::IsTargetVar(uint32_t varId) {
  if (seen_non_target_vars_.find(varId) != seen_non_target_vars_.end())
    return false;
  if (seen_target_vars_.find(varId) != seen_target_vars_.end())
    return true;
  const ir::Instruction* varInst =
    def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
    def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) !=
    SpvStorageClassFunction) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
    varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeId);
  ir::Instruction* varPteTypeInst =
    def_use_mgr_->id_to_defs().find(varPteTypeId)->second;
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

void LocalSingleStoreElimPass::SingleStoreAnalyze(ir::Function* func) {
  ssa_var2store_.clear();
  non_ssa_vars_.clear();
  store2idx_.clear();
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii, ++instIdx) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (non_ssa_vars_.find(varId) != non_ssa_vars_.end())
          continue;
        if (ptrInst->opcode() == SpvOpAccessChain) {
          non_ssa_vars_.insert(varId);
          ssa_var2store_.erase(varId);
          continue;
        }
        // Verify target type and function storage class
        if (!IsTargetVar(varId)) {
          non_ssa_vars_.insert(varId);
          continue;
        }
        // If already stored, disqualify it
        if (ssa_var2store_.find(varId) != ssa_var2store_.end()) {
          non_ssa_vars_.insert(varId);
          ssa_var2store_.erase(varId);
          continue;
        }
        // Remember iterator of variable's store and it's
        // ordinal position in function
        ssa_var2store_[varId] = &*ii;
        store2idx_[&*ii] = instIdx;
      } break;
      case SpvOpFunctionCall: {
        // For now, empty SSA variable set and terminate analysis
        // TODO(): Add logic to more optimally handle function calls.
        ssa_var2store_.clear();
        return;
      } break;
      default:
        break;
      } // switch
    }
  }
}

void LocalSingleStoreElimPass::DeleteIfUseless(ir::Instruction* inst) {
  const uint32_t resId = inst->result_id();
  assert(resId != 0);
  analysis::UseList* uses = def_use_mgr_->GetUses(resId);
  if (uses == nullptr)
    def_use_mgr_->KillInst(inst);
}

void LocalSingleStoreElimPass::ReplaceAndDeleteLoad(ir::Instruction* loadInst,
                                      uint32_t replId,
                                      ir::Instruction* ptrInst) {
  const uint32_t loadId = loadInst->result_id();
  (void) def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
  // remove load instruction
  def_use_mgr_->KillInst(loadInst);
  // if access chain, see if it can be removed as well
  if (ptrInst->opcode() == SpvOpAccessChain) {
    DeleteIfUseless(ptrInst);
  }
}

bool LocalSingleStoreElimPass::SingleStoreProcess(ir::Function* func) {
  bool modified = false;
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii, ++instIdx) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      uint32_t varId;
      ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
      // Skip access chain loads
      if (ptrInst->opcode() == SpvOpAccessChain)
        continue;
      assert(ptrInst->opcode() == SpvOpVariable);
      const auto vsi = ssa_var2store_.find(varId);
      if (vsi == ssa_var2store_.end())
        continue;
      if (non_ssa_vars_.find(varId) != non_ssa_vars_.end())
        continue;
      // Use store value as replacement id
      uint32_t replId = vsi->second->GetSingleWordInOperand(kSpvStoreValId);
      // store must dominate load
      if (instIdx < store2idx_[vsi->second])
        continue;
      // replace all instances of the load's id with the SSA value's id
      ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
      modified = true;
    }
  }
  return modified;
}

bool LocalSingleStoreElimPass::HasLoads(uint32_t varId) {
  analysis::UseList* uses = def_use_mgr_->GetUses(varId);
  if (uses == nullptr)
    return false;
  for (auto u : *uses) {
    if (u.inst->opcode() == SpvOpAccessChain) {
      uint32_t cid = u.inst->result_id();
      analysis::UseList* cuses = def_use_mgr_->GetUses(cid);
      if (cuses == nullptr)
        continue;
      for (auto cu : *cuses)
        if (cu.inst->opcode() == SpvOpLoad)
          return true;
    }
    else if (u.inst->opcode() == SpvOpLoad)
      return true;
  }
  return false;
}

bool LocalSingleStoreElimPass::IsLiveVar(uint32_t varId) {
  // non-function scope vars are live
  const ir::Instruction* varInst =
      def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
      def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
  return HasLoads(varId);
}

bool LocalSingleStoreElimPass::IsLiveStore(ir::Instruction* storeInst) {
  // get store's variable
  uint32_t varId;
  (void) GetPtr(storeInst, &varId);
  return IsLiveVar(varId);
}

void LocalSingleStoreElimPass::DCEInst(ir::Instruction* inst) {
  std::queue<ir::Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    ir::Instruction* di = deadInsts.front();
    // Don't delete labels
    if (di->opcode() == SpvOpLabel) {
      deadInsts.pop();
      continue;
    }
    std::queue<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) {
      ids.push(*iid);
    });
    uint32_t varId = 0;
    if (di->opcode() == SpvOpLoad)
      (void) GetPtr(di, &varId);
    def_use_mgr_->KillInst(di);
    while (!ids.empty()) {
      uint32_t id = ids.front();
      analysis::UseList* uses = def_use_mgr_->GetUses(id);
      if (uses == nullptr)
        deadInsts.push(def_use_mgr_->GetDef(id));
      ids.pop();
    }
    // if a load was deleted and it was the variable's
    // last load, add all its stores to dead queue
    if (varId != 0 && !IsLiveVar(varId)) {
      analysis::UseList* uses = def_use_mgr_->GetUses(varId);
      if (uses != nullptr) {
        for (auto u : *uses) {
          if (u.inst->opcode() == SpvOpAccessChain) {
            uint32_t cid = u.inst->result_id();
            analysis::UseList* cuses = def_use_mgr_->GetUses(cid);
            if (cuses != nullptr) {
              for (auto cu : *cuses)
                if (cu.inst->opcode() == SpvOpStore)
                  deadInsts.push(cu.inst);
            }
          }
          else if (u.inst->opcode() == SpvOpStore)
            deadInsts.push(u.inst);
        }
      }
    }
    deadInsts.pop();
  }
}

bool LocalSingleStoreElimPass::SingleStoreDCE() {
  bool modified = false;
  for (auto v : ssa_var2store_) {
    // check that it hasn't already been DCE'd
    if (v.second->opcode() != SpvOpStore)
      continue;
    if (non_ssa_vars_.find(v.first) != non_ssa_vars_.end())
      continue;
    if (!IsLiveStore(v.second)) {
      DCEInst(v.second);
      modified = true;
    }
  }
  return modified;
}

bool LocalSingleStoreElimPass::LocalSingleStoreElim(ir::Function* func) {
  bool modified = false;
  SingleStoreAnalyze(func);
  if (ssa_var2store_.empty())
    return false;
  modified |= SingleStoreProcess(func);
  modified |= SingleStoreDCE();
  return modified;
}

void LocalSingleStoreElimPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_)
    id2function_[fn.result_id()] = &fn;

  // Initialize Target Type Caches
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();

  // TODO: Reuse def/use (and other state) from previous passes
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize next unused Id
  next_id_ = module_->id_bound();
};

Pass::Status LocalSingleStoreElimPass::ProcessImpl() {
  bool modified = false;

  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || LocalSingleStoreElim(fn);
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalSingleStoreElimPass::LocalSingleStoreElimPass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status LocalSingleStoreElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

