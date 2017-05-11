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

#include "ssamem_pass.h"
#include "iterator.h"

/*
#define SPV_FUNCTION_CALL_FUNCTION_ID 2
#define SPV_FUNCTION_CALL_ARGUMENT_ID 3
#define SPV_FUNCTION_PARAMETER_RESULT_ID 1
#define SPV_STORE_POINTER_ID 0
#define SPV_STORE_OBJECT_ID 1
#define SPV_RETURN_VALUE_ID 0
#define SPV_TYPE_POINTER_STORAGE_CLASS 1
#define SPV_TYPE_POINTER_TYPE_ID 2
*/

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvFunctionCallFunctionId = 0;
static const int kSpvDecorationTargetId = 0;
static const int kSpvDecorationDecoration = 1;
static const int kSpvDecorationLinkageType = 3;
static const int kSpvStorePtrId = 0;
static const int kSpvStoreValId = 1;
static const int kSpvLoadPtrId = 0;
static const int kSpvAccessChainPtrId = 0;
static const int kSpvTypePointerStorageClass = 0;
static const int kSpvTypePointerTypeId = 1;
static const int kSpvConstantValue = 0;
static const int kSpvExtractCompositeId = 0;
static const int kSpvExtractIdx0 = 1;
static const int kSpvInsertObjectId = 0;
static const int kSpvInsertCompositeId = 1;
static const int kSpvBranchCondConditionalId = 0;
static const int kSpvBranchCondTrueLabId = 1;
static const int kSpvBranchCondFalseLabId = 2;
static const int kSpvSelectionMergeMergeBlockId = 0;
static const int kSpvPhiVal0Id = 0;
static const int kSpvPhiLab0Id = 1;
static const int kSpvPhiVal1Id = 2;
static const int kSpvLoopMergeMergeBlockId = 0;

namespace spvtools {
namespace opt {

bool SSAMemPass::IsMathType(const ir::Instruction* typeInst) {
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

bool SSAMemPass::IsTargetType(const ir::Instruction* typeInst) {
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

ir::Instruction* SSAMemPass::GetPtr(ir::Instruction* ip, uint32_t* varId) {
  const uint32_t ptrId = ip->opcode() == SpvOpStore ?
      ip->GetInOperand(kSpvStorePtrId).words[0] :
      ip->GetInOperand(kSpvLoadPtrId).words[0];
  ir::Instruction* ptrInst =
    def_use_mgr_->id_to_defs().find(ptrId)->second;
  *varId = ptrInst->opcode() == SpvOpAccessChain ?
    ptrInst->GetInOperand(kSpvAccessChainPtrId).words[0] :
    ptrId;
  return ptrInst;
}

bool SSAMemPass::IsUniformVar(uint32_t varId) {
  const ir::Instruction* varInst =
    def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
    def_use_mgr_->id_to_defs().find(varTypeId)->second;
  return varTypeInst->GetInOperand(kSpvTypePointerStorageClass).words[0] ==
    SpvStorageClassUniform ||
    varTypeInst->GetInOperand(kSpvTypePointerStorageClass).words[0] ==
    SpvStorageClassUniformConstant;
}

bool SSAMemPass::IsTargetVar(uint32_t varId) {
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
  if (varTypeInst->GetInOperand(kSpvTypePointerStorageClass).words[0] !=
    SpvStorageClassFunction) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
    varTypeInst->GetInOperand(kSpvTypePointerTypeId).words[0];
  ir::Instruction* varPteTypeInst =
    def_use_mgr_->id_to_defs().find(varPteTypeId)->second;
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

void SSAMemPass::SingleStoreAnalyze(ir::Function* func) {
  ssa_var2store_.clear();
  non_ssa_vars_.clear();
  store2idx_.clear();
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii, ++instIdx) {
      if (ii->opcode() != SpvOpStore)
        continue;
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
    }
  }
}

void SSAMemPass::DeleteIfUseless(ir::Instruction* inst) {
  const uint32_t resId = inst->result_id();
  assert(resId != 0);
  analysis::UseList* uses = def_use_mgr_->GetUses(resId);
  if (uses == nullptr)
    def_use_mgr_->KillInst(inst);
}

void SSAMemPass::ReplaceAndDeleteLoad(ir::Instruction* loadInst,
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

bool SSAMemPass::SingleStoreProcess(ir::Function* func) {
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
      uint32_t replId = vsi->second->GetInOperand(kSpvStoreValId).words[0];
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

bool SSAMemPass::HasLoads(uint32_t varId) {
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

bool SSAMemPass::IsLiveVar(uint32_t varId) {
  // non-function scope vars are live
  const ir::Instruction* varInst =
      def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
      def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetInOperand(kSpvTypePointerStorageClass).words[0] !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
  return HasLoads(varId);
}

bool SSAMemPass::IsLiveStore(ir::Instruction* storeInst) {
  // get store's variable
  uint32_t varId;
  (void) GetPtr(storeInst, &varId);
  return IsLiveVar(varId);
}

void SSAMemPass::DCEInst(ir::Instruction* inst) {
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

bool SSAMemPass::SingleStoreDCE() {
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

bool SSAMemPass::LocalSingleStoreElim(ir::Function* func) {
  bool modified = false;
  SingleStoreAnalyze(func);
  modified |= SingleStoreProcess(func);
  modified |= SingleStoreDCE();
  return modified;
}

bool SSAMemPass::LocalSingleBlockElim(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    var2store_.clear();
    var2load_.clear();
    pinned_vars_.clear();
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          continue;
        // Register the store
        if (ptrInst->opcode() == SpvOpVariable) {
          // if not pinned, look for WAW
          if (pinned_vars_.find(varId) == pinned_vars_.end()) {
            auto si = var2store_.find(varId);
            if (si != var2store_.end()) {
              def_use_mgr_->KillInst(si->second);
            }
          }
          var2store_[varId] = &*ii;
        }
        else {
          assert(ptrInst->opcode() == SpvOpAccessChain);
          var2store_.erase(varId);
        }
        pinned_vars_.erase(varId);
        var2load_.erase(varId);
      } break;
      case SpvOpLoad: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          continue;
        // Look for previous store or load
        uint32_t replId = 0;
        if (ptrInst->opcode() == SpvOpVariable) {
          auto si = var2store_.find(varId);
          if (si != var2store_.end()) {
            replId = si->second->GetInOperand(kSpvStoreValId).words[0];
          }
          else {
            auto li = var2load_.find(varId);
            if (li != var2load_.end()) {
              replId = li->second->result_id();
            }
          }
        }
        if (replId != 0) {
          // replace load's result id and delete load
          ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
          modified = true;
        }
        else {
          if (ptrInst->opcode() == SpvOpVariable)
            var2load_[varId] = &*ii;  // register load
          pinned_vars_.insert(varId);
        }
      } break;
      default:
        break;
      }
    }
    // Go back and delete useless stores in block
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore)
        continue;
      if (IsLiveStore(&*ii))
        continue;
      DCEInst(&*ii);
    }
  }
  return modified;
}

bool SSAMemPass::FuncDCE(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        (void)GetPtr(&*ii, &varId);
        if (IsLiveVar(varId))
          break;
        DCEInst(&*ii);
        modified = true;
      } break;
      default: {
        uint32_t rId = ii->result_id();
        if (rId == 0)
          break;
        analysis::UseList* uses = def_use_mgr_->GetUses(rId);
        if (uses != nullptr)
          break;
        DCEInst(&*ii);
        modified = true;
      } break;
      }
    }
  }
  return modified;
}

uint32_t SSAMemPass::GetPteTypeId(const ir::Instruction* ptrInst) {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const ir::Instruction* ptrTypeInst =
      def_use_mgr_->id_to_defs().find(ptrTypeId)->second;
  return ptrTypeInst->GetInOperand(kSpvTypePointerTypeId).words[0];
}

void SSAMemPass::GenACLoadRepl(const ir::Instruction* ptrInst,
  std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
  uint32_t& resultId) {

  // Build and append Load
  const uint32_t ldResultId = TakeNextId();
  const uint32_t varId =
    ptrInst->GetInOperand(kSpvAccessChainPtrId).words[0];
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varPteTypeId = GetPteTypeId(varInst);
  std::vector<ir::Operand> load_in_operands;
  load_in_operands.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{varId}));
  std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(SpvOpLoad,
    varPteTypeId, ldResultId, load_in_operands));
  def_use_mgr_->AnalyzeInstDefUse(&*newLoad);
  newInsts.emplace_back(std::move(newLoad));

  // Build and append Extract
  const uint32_t extResultId = TakeNextId();
  const uint32_t ptrPteTypeId = GetPteTypeId(ptrInst);
  std::vector<ir::Operand> ext_in_opnds;
  ext_in_opnds.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{ldResultId}));
  uint32_t iidIdx = 0;
  ptrInst->ForEachInId([&iidIdx, &ext_in_opnds, this](const uint32_t *iid) {
    if (iidIdx > 0) {
      const ir::Instruction* cInst = def_use_mgr_->GetDef(*iid);
      uint32_t val = cInst->GetInOperand(kSpvConstantValue).words[0];
      ext_in_opnds.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          std::initializer_list<uint32_t>{val}));
    }
    ++iidIdx;
  });
  std::unique_ptr<ir::Instruction> newExt(new ir::Instruction(
    SpvOpCompositeExtract, ptrPteTypeId, extResultId, ext_in_opnds));
  def_use_mgr_->AnalyzeInstDefUse(&*newExt);
  newInsts.emplace_back(std::move(newExt));
  resultId = extResultId;
}

void SSAMemPass::GenACStoreRepl(const ir::Instruction* ptrInst,
  uint32_t valId,
  std::vector<std::unique_ptr<ir::Instruction>>& newInsts) {

  // Build and append Load
  const uint32_t ldResultId = TakeNextId();
  const uint32_t varId =
    ptrInst->GetInOperand(kSpvAccessChainPtrId).words[0];
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varPteTypeId = GetPteTypeId(varInst);
  std::vector<ir::Operand> load_in_operands;
  load_in_operands.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{varId}));
  std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(SpvOpLoad,
    varPteTypeId, ldResultId, load_in_operands));
  def_use_mgr_->AnalyzeInstDefUse(&*newLoad);
  newInsts.emplace_back(std::move(newLoad));

  // Build and append Insert
  const uint32_t insResultId = TakeNextId();
  std::vector<ir::Operand> ins_in_opnds;
  ins_in_opnds.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{valId}));
  ins_in_opnds.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{ldResultId}));
  uint32_t iidIdx = 0;
  ptrInst->ForEachInId([&iidIdx, &ins_in_opnds, this](const uint32_t *iid) {
    if (iidIdx > 0) {
      const ir::Instruction* cInst = def_use_mgr_->GetDef(*iid);
      uint32_t val = cInst->GetInOperand(kSpvConstantValue).words[0];
      ins_in_opnds.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          std::initializer_list<uint32_t>{val}));
    }
    ++iidIdx;
  });
  std::unique_ptr<ir::Instruction> newIns(new ir::Instruction(
    SpvOpCompositeInsert, varPteTypeId, insResultId, ins_in_opnds));
  def_use_mgr_->AnalyzeInstDefUse(&*newIns);
  newInsts.emplace_back(std::move(newIns));

  // Build and append Store
  std::vector<ir::Operand> store_in_operands;
  store_in_operands.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{varId}));
  store_in_operands.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{insResultId}));
  std::unique_ptr<ir::Instruction> newStore(new ir::Instruction(SpvOpStore,
      0, 0, store_in_operands));
  def_use_mgr_->AnalyzeInstDefUse(&*newStore);
  newInsts.emplace_back(std::move(newStore));
}

bool SSAMemPass::IsConstantIndexAccessChain(ir::Instruction* acp) {
  uint32_t inIdx = 0;
  uint32_t nonConstCnt = 0;
  acp->ForEachInId([&inIdx, &nonConstCnt, this](uint32_t* tid) {
    if (inIdx > 0) {
      ir::Instruction* opInst = def_use_mgr_->GetDef(*tid);
      if (opInst->opcode() != SpvOpConstant) ++nonConstCnt;
    }
    ++inIdx;
  });
  return nonConstCnt == 0;
}

bool SSAMemPass::LocalAccessChainConvert(ir::Function* func) {
  // rule out variables accessed with non-constant indices
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore:
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!IsTargetVar(varId))
          break;
        if (!IsConstantIndexAccessChain(ptrInst)) {
          seen_non_target_vars_.insert(varId);
          seen_target_vars_.erase(varId);
          break;
        }
      } break;
      default:
        break;
      }
    }
  }
  // replace ACs of all targeted variables
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!IsTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        uint32_t replId;
        GenACLoadRepl(ptrInst, newInsts, replId);
        ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
        ++ii;
        ii = ii.MoveBefore(newInsts);
        ++ii;
        modified = true;
      } break;
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!IsTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        uint32_t valId = ii->GetInOperand(kSpvStoreValId).words[0];
        GenACStoreRepl(ptrInst, valId, newInsts);
        def_use_mgr_->KillInst(&*ii);
        DeleteIfUseless(ptrInst);
        ++ii;
        ii = ii.MoveBefore(newInsts);
        ++ii;
        ++ii;
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  return modified;
}

bool SSAMemPass::UniformAccessChainConvert(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      uint32_t varId;
      ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
      if (ptrInst->opcode() != SpvOpAccessChain)
        continue;
      if (!IsUniformVar(varId))
        continue;
      if (!IsConstantIndexAccessChain(ptrInst))
        continue;
      std::vector<std::unique_ptr<ir::Instruction>> newInsts;
      uint32_t replId;
      GenACLoadRepl(ptrInst, newInsts, replId);
      ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
      ++ii;
      ii = ii.MoveBefore(newInsts);
      ++ii;
      modified = true;
    }
  }
  return modified;
}

bool SSAMemPass::CommonUniformLoadElimination(ir::Function* func) {
  bool modified = false;
  // Find insertion point in first block to copy all non-dominating
  // loads.
  auto insertItr = func->begin()->begin();
  while (insertItr->opcode() == SpvOpVariable || insertItr->opcode() == SpvOpNop)
    ++insertItr;
  uint32_t mergeBlockId = 0;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    // Check if we are exiting control flow
    if (mergeBlockId == bi->GetLabelId())
      mergeBlockId = 0;
    // Check if we are enterinig loop
    if (mergeBlockId == 0 && IsLoopHeader(&*bi))
      mergeBlockId = GetMergeBlkId(&*bi);
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      uint32_t varId;
      ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
      if (ptrInst->opcode() == SpvOpAccessChain)
        continue;
      if (!IsUniformVar(varId))
        continue;
      uint32_t replId;
      const auto uItr = uniform2load_id_.find(varId);
      if (uItr != uniform2load_id_.end()) {
        replId = uItr->second;
      }
      else {
        if (mergeBlockId == 0) {
          // Load is in dominating block; just remember it
          uniform2load_id_[varId] = ii->result_id();
          continue;
        }
        else {
          // Copy load into first block and remember it
          replId = TakeNextId();
          std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(SpvOpLoad,
            ii->type_id(), replId, {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {varId}}}));
          def_use_mgr_->AnalyzeInstDefUse(&*newLoad);
          insertItr = insertItr.InsertBefore(std::move(newLoad));
          ++insertItr;
          uniform2load_id_[varId] = replId;
        }
      }
      ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
      modified = true;
    }
    // Check if we are entering select
    if (mergeBlockId == 0)
      mergeBlockId = GetMergeBlkId(&*bi);
  }
  return modified;
}

bool SSAMemPass::CommonExtractElimination(ir::Function* func) {
  // Find all composite ids with duplicate extracts.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpCompositeExtract)
        continue;
      if (ii->NumInOperands() > 2)
        continue;
      uint32_t compId = ii->GetSingleWordInOperand(kSpvExtractCompositeId);
      uint32_t idx = ii->GetSingleWordInOperand(kSpvExtractIdx0);
      comp2idx2inst_[compId][idx].push_back(&*ii);
    }
  }
  // For all defs of ids with duplicate extracts, insert new extracts
  // after def, and replace and delete old extracts
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      const auto cItr = comp2idx2inst_.find(ii->result_id());
      if (cItr == comp2idx2inst_.end())
        continue;
      for (auto idxItr : cItr->second) {
        if (idxItr.second.size() < 2)
          continue;
        uint32_t replId = TakeNextId();
        std::unique_ptr<ir::Instruction> newExtract(new ir::Instruction(*idxItr.second.front()));
        newExtract->SetResultId(replId);
        def_use_mgr_->AnalyzeInstDefUse(&*newExtract);
        ++ii;
        ii = ii.InsertBefore(std::move(newExtract));
        for (auto instItr : idxItr.second) {
          (void)def_use_mgr_->ReplaceAllUsesWith(instItr->result_id(), replId);
          def_use_mgr_->KillInst(instItr);
        }
        modified = true;
      }
    }
  }
  return modified;
}

bool SSAMemPass::IsStructured(ir::Function* func) {
  // TODO(greg-lunarg)
  // All control flow is structured
  // All non-nested headers have non-backedge predecessor
  (void)func;
  return true;
}

uint32_t SSAMemPass::GetMergeBlkId(ir::BasicBlock* block_ptr) {
  auto iItr = block_ptr->end();
  --iItr;
  if (iItr == block_ptr->begin())
    return 0;
  --iItr;
  if (iItr->opcode() == SpvOpLoopMerge)
    return iItr->GetSingleWordInOperand(kSpvLoopMergeMergeBlockId);
  else if (iItr->opcode() == SpvOpSelectionMerge)
    return iItr->GetSingleWordInOperand(kSpvSelectionMergeMergeBlockId);
  else
    return 0;
}

bool SSAMemPass::IsLoopHeader(ir::BasicBlock* block_ptr) {
  auto iItr = block_ptr->end();
  --iItr;
  if (iItr == block_ptr->begin())
    return false;
  --iItr;
  return iItr->opcode() == SpvOpLoopMerge;
}

void SSAMemPass::SSABlockInitSinglePred(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  uint32_t predLabel = label2preds_[label].front();
  assert(visitedBlocks.find(predLabel) != visitedBlocks.end());
  label2ssa_map_[label] = label2ssa_map_[predLabel];
}

void SSAMemPass::InitSSARewrite(ir::Function& func) {
  // Init predecessors
  label2preds_.clear();
  for (auto& blk : func) {
    uint32_t blkId = blk.GetLabelId();
    blk.ForEachSucc([&blkId, this](uint32_t sbid) {
      label2preds_[sbid].push_back(blkId);
    });
  }
  // Init IsLiveAfter
  block2loop_.clear();
  loop2last_block_.clear();
  block2ord_.clear();
  var2last_load_block_.clear();
  uint32_t blockOrd = 0;
  uint32_t outerLoopCount = 0;
  uint32_t outerLoopOrd = 0;
  uint32_t nextMergeBlockId = 0;
  uint32_t lastBlkId = 0;
  for (auto& blk : func) {
    uint32_t blkId = blk.GetLabelId();
    block2ord_[blkId] = blockOrd;
    if (blk.GetLabelId() == nextMergeBlockId) {
      loop2last_block_[outerLoopOrd] = lastBlkId;
      outerLoopOrd = 0;
    }
    if (outerLoopOrd == 0 && IsLoopHeader(&blk)) {
      ++outerLoopCount;
      outerLoopOrd = outerLoopCount;
      nextMergeBlockId =
          blk.begin()->GetSingleWordInOperand(kSpvLoopMergeMergeBlockId);
    }
    block2loop_[blkId] = outerLoopOrd;
    for (auto& inst : blk) {
      if (inst.opcode() != SpvOpLoad)
        continue;
      uint32_t varId;
      (void) GetPtr(&inst, &varId);
      if (!IsTargetVar(varId))
        break;
      var2last_load_block_[varId] = blkId;
    }
    lastBlkId = blkId;
    ++blockOrd;
  }
  // Compute the last live block for each variable
  for (auto& var_blk : var2last_load_block_) {
    uint32_t varId = var_blk.first;
    uint32_t lastLiveBlkId = var_blk.second;
    uint32_t loopOrd = block2loop_[lastLiveBlkId];
    if (loopOrd != 0)
       lastLiveBlkId = loop2last_block_[loopOrd];
    var2last_live_block_[varId] = lastLiveBlkId;
  }
}

bool SSAMemPass::IsLiveAfter(uint32_t var_id, uint32_t label) {
  // TODO(): This code currently causes bad images. It seems to
  // be missing some cases. Retry when inlining of early returns
  // is fixed. Until then this routine will cause correct, but
  // possibly usused code to be generated. A subsequent DCE pass
  // would likely eliminate this code, so it is a question of
  // compile-time efficiency.
  //auto isLiveItr = var2last_live_block_.find(var_id);
  //if (isLiveItr == var2last_live_block_.end())
  //  return false;
  //return block2ord_[label] <= block2ord_[isLiveItr->second];
  (void)var_id;
  (void)label;
  return true;
}

uint32_t SSAMemPass::Type2Undef(uint32_t type_id) {
  const auto uitr = type2undefs_.find(type_id);
  if (uitr != type2undefs_.end())
    return uitr->second;
  uint32_t undefId = TakeNextId();
  std::unique_ptr<ir::Instruction> undef_inst(
    new ir::Instruction(SpvOpUndef, type_id, undefId, {}));
  def_use_mgr_->AnalyzeInstDefUse(&*undef_inst);
  module_->AddGlobalValue(std::move(undef_inst));
  type2undefs_[type_id] = undefId;
  return undefId;
}

void SSAMemPass::SSABlockInitLoopHeader(
    ir::UptrVectorIterator<ir::BasicBlock> block_itr) {
  uint32_t label = block_itr->GetLabelId();
  // Determine backedge label.
  uint32_t backLabel = 0;
  for (uint32_t predLabel : label2preds_[label])
    if (visitedBlocks.find(predLabel) == visitedBlocks.end()) {
      assert(backLabel == 0);
      backLabel = predLabel;
      break;
    }
  assert(backLabel != 0);
  // Determine merge block.
  auto mergeInst = block_itr->begin();
  uint32_t mergeLabel = mergeInst->GetSingleWordInOperand(
      kSpvLoopMergeMergeBlockId);
  // Collect all live variables and a default value for each across all
  // non-backedge predecesors.
  std::unordered_map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    for (auto var_val : label2ssa_map_[predLabel]) {
      uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // Add all stored variables in loop. Set their default value id to zero.
  for (auto bi = block_itr; bi->GetLabelId() != mergeLabel; ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore)
        continue;
      uint32_t varId;
      (void) GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId))
        break;
      liveVars[varId] = 0;
    }
  }
  // Insert phi for all live variables that require them. All variables
  // defined in loop require a phi. Otherwise all variables
  // with differing predecessor values require a phi.
  auto insertItr = block_itr->begin();
  for (auto var_val : liveVars) {
    uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool needsPhi = false;
    if (val0Id != 0) {
      for (uint32_t predLabel : label2preds_[label]) {
        // Skip back edge predecessor.
        if (predLabel == backLabel)
          continue;
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        // Missing values do not cause difference
        if (var_val_itr == label2ssa_map_[predLabel].end())
          continue;
        if (var_val_itr->second != val0Id) {
          needsPhi = true;
          break;
        }
      }
    }
    else {
      needsPhi = true;
    }
    // If val is the same for all predecessors, enter it in map
    if (!needsPhi) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map. For live back edge predecessor,
    // use the variable id. We will patch this after visiting back
    // edge predecessor. For predessors that do not define a value,
    // use undef.
    std::vector<ir::Operand> phi_in_operands;
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      uint32_t valId;
      if (predLabel == backLabel) {
        if (val0Id == 0)
          valId = varId;
        else
          valId = Type2Undef(typeId);
      }
      else {
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        if (var_val_itr == label2ssa_map_[predLabel].end())
          valId = Type2Undef(typeId);
        else
          valId = var_val_itr->second;
      }
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    // Only analyze the phi define now; analyze the phi uses after the
    // phi backedge predecessor value is patched.
    def_use_mgr_->AnalyzeInstDef(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({ varId, phiId });
  }
}

void SSAMemPass::SSABlockInitSelectMerge(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  // Collect all live variables and a default value for each across all
  // predecesors.
  std::unordered_map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    assert(visitedBlocks.find(predLabel) != visitedBlocks.end());
    for (auto var_val : label2ssa_map_[predLabel]) {
      uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // For each live variable, look for a difference in values across
  // predecessors that would require a phi and insert one.
  auto insertItr = block_ptr->begin();
  for (auto var_val : liveVars) {
    uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool differs = false;
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // Missing values cause a difference
      if (var_val_itr == label2ssa_map_[predLabel].end()) {
        differs = true;
        break;
      }
      if (var_val_itr->second != val0Id) {
        differs = true;
        break;
      }
    }
    // If val is the same for all predecessors, enter it in SSA map
    if (!differs) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map
    std::vector<ir::Operand> phi_in_operands;
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // If variable not defined on this path, use undef
      uint32_t valId = (var_val_itr != label2ssa_map_[predLabel].end()) ?
          var_val_itr->second : Type2Undef(typeId);
      phi_in_operands.push_back(
          ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    def_use_mgr_->AnalyzeInstDefUse(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({varId, phiId});
  }
}

void SSAMemPass::SSABlockInit(ir::UptrVectorIterator<ir::BasicBlock> block_itr) {
  size_t numPreds = label2preds_[block_itr->GetLabelId()].size();
  if (numPreds == 0)
    return;
  if (numPreds == 1)
    SSABlockInitSinglePred(&*block_itr);
  else if (IsLoopHeader(&*block_itr))
    SSABlockInitLoopHeader(block_itr);
  else
    SSABlockInitSelectMerge(&*block_itr);
}

void SSAMemPass::PatchPhis(uint32_t header_id, uint32_t back_id) {
  ir::BasicBlock* header = label2block_[header_id];
  auto phiItr = header->begin();
  for (; phiItr->opcode() == SpvOpPhi; ++phiItr) {
    uint32_t cnt = 0;
    uint32_t idx;
    phiItr->ForEachInId([&cnt,&back_id,&idx](uint32_t* iid) {
      if (cnt % 2 == 1 && *iid == back_id) idx = cnt - 1;
      ++cnt;
    });
    // Only patch operands that are in the backedge predecessor map
    uint32_t varId = phiItr->GetSingleWordInOperand(idx);
    const auto valItr = label2ssa_map_[back_id].find(varId);
    if (valItr != label2ssa_map_[back_id].end()) {
      phiItr->SetInOperand(idx, { valItr->second });
      // Analyze uses now that they are complete
      def_use_mgr_->AnalyzeInstUse(&*phiItr);
    }
  }
}

bool SSAMemPass::LocalSSARewrite(ir::Function* func) {
  if (!IsStructured(func))
    return false;
  InitSSARewrite(*func);
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    // Initialize this block's SSA map using predecessor's maps.
    SSABlockInit(bi);
    uint32_t label = bi->GetLabelId();
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        uint32_t valId = ii->GetInOperand(kSpvStoreValId).words[0];
        // Register new stored value for the variable
        label2ssa_map_[label][varId] = valId;
      } break;
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        // If variable is not defined, use undef
        uint32_t replId = 0;
        const auto ssaItr = label2ssa_map_.find(label);
        if (ssaItr != label2ssa_map_.end()) {
          const auto valItr = ssaItr->second.find(varId);
          if (valItr != ssaItr->second.end())
            replId = valItr->second;
        }
        if (replId == 0) {
          uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
          replId = Type2Undef(typeId);
        }
        // Replace load's id with the last stored value id
        // and delete load.
        const uint32_t loadId = ii->result_id();
        (void)def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
        def_use_mgr_->KillInst(&*ii);
        modified = true;
      } break;
      default: {
      } break;
      }
    }
    visitedBlocks.insert(label);
    // Look for successor backedge and patch phis in loop header
    // if found.
    uint32_t header = 0;
    bi->ForEachSucc([&header,this](uint32_t succ) {
      if (visitedBlocks.find(succ) == visitedBlocks.end()) return;
      assert(header == 0);
      header = succ;
    });
    if (header != 0)
      PatchPhis(header, label);
  }
  // Remove all target variable stores.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore)
        continue;
      uint32_t varId;
      (void) GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId))
        break;
      assert(!HasLoads(varId));
      DCEInst(&*ii);
      modified = true;
    }
  }
  return modified;
}


bool SSAMemPass::SSAMemExtInsMatch(ir::Instruction* extInst,
    ir::Instruction* insInst) {
  if (extInst->NumInOperands() != insInst->NumInOperands() - 1)
    return false;
  uint32_t numIdx = extInst->NumInOperands() - 1;
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetInOperand(i + 1).words[0] !=
        insInst->GetInOperand(i + 2).words[0])
      return false;
  return true;
}

bool SSAMemPass::SSAMemExtInsConflict(ir::Instruction* extInst,
    ir::Instruction* insInst) {
  if (extInst->NumInOperands() == insInst->NumInOperands() - 1)
    return false;
  uint32_t extNumIdx = extInst->NumInOperands() - 1;
  uint32_t insNumIdx = insInst->NumInOperands() - 2;
  uint32_t numIdx = extNumIdx < insNumIdx ? extNumIdx : insNumIdx;
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetInOperand(i + 1).words[0] !=
        insInst->GetInOperand(i + 2).words[0])
      return false;
  return true;
}

bool SSAMemPass::InsertExtractElim(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpCompositeExtract: {
        uint32_t cid = ii->GetInOperand(kSpvExtractCompositeId).words[0];
        ir::Instruction* cinst = def_use_mgr_->GetDef(cid);
        uint32_t replId = 0;
        while (cinst->opcode() == SpvOpCompositeInsert) {
          if (SSAMemExtInsConflict(&*ii, cinst))
            break;
          if (SSAMemExtInsMatch(&*ii, cinst)) {
            replId = cinst->GetInOperand(kSpvInsertObjectId).words[0];
            break;
          }
          cid = cinst->GetInOperand(kSpvInsertCompositeId).words[0];
          cinst = def_use_mgr_->GetDef(cid);
        }
        if (replId == 0)
          break;
        const uint32_t extId = ii->result_id();
        (void)def_use_mgr_->ReplaceAllUsesWith(extId, replId);
        def_use_mgr_->KillInst(&*ii);
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  return modified;
}

bool SSAMemPass::InsertCycleBreak(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (ptrInst->opcode() != SpvOpVariable)
          break;
        if (!IsTargetVar(varId))
          break;
        analysis::UseList* vuses = def_use_mgr_->GetUses(varId);
        if (vuses->size() > 2)
          break;
        uint32_t valId = ii->GetInOperand(kSpvStoreValId).words[0];
        ir::Instruction* vinst = def_use_mgr_->GetDef(valId);
        uint32_t intermediate = false;
        while (vinst->opcode() == SpvOpCompositeInsert) {
          if (intermediate) {
            const uint32_t resId = vinst->result_id();
            analysis::UseList* uses = def_use_mgr_->GetUses(resId);
            if (uses->size() > 1)
              break;
          }
          valId = vinst->GetInOperand(kSpvInsertCompositeId).words[0];
          vinst = def_use_mgr_->GetDef(valId);
          intermediate = true;
        }
        if (vinst->opcode() != SpvOpLoad)
          break;
        uint32_t lvarId = vinst->GetInOperand(kSpvLoadPtrId).words[0];
        if (lvarId != varId)
          break;
        const uint32_t resId = vinst->result_id();
        analysis::UseList* uses = def_use_mgr_->GetUses(resId);
        if (uses->size() > 1)
          break;
        DCEInst(&*ii);
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  if (modified) {
    // look for loads with no stores and replace with undef
    for (auto bi = func->begin(); bi != func->end(); ++bi) {
      for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
        switch (ii->opcode()) {
        case SpvOpLoad: {
          uint32_t varId;
          ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
          if (ptrInst->opcode() != SpvOpVariable)
            break;
          if (!IsTargetVar(varId))
            break;
          analysis::UseList* vuses = def_use_mgr_->GetUses(varId);
          if (vuses->size() == 1) {
            uint32_t undefId = TakeNextId();
            uint32_t typeId = ii->type_id();
            std::unique_ptr<ir::Instruction> undef_inst(
              new ir::Instruction(SpvOpUndef, typeId, undefId, {}));
            def_use_mgr_->AnalyzeInstDefUse(&*undef_inst);
            ii = ii.InsertBefore(std::move(undef_inst));
            ++ii; // back to load
            assert(ii->opcode() == SpvOpLoad);
            uint32_t loadId = ii->result_id();
            (void)def_use_mgr_->ReplaceAllUsesWith(loadId, undefId);
            DCEInst(&*ii);
          }
        } break;
        default:
          break;
        }
      }
    }
  }
  return modified;
}

void SSAMemPass::SSAMemGetConstCondition(
    uint32_t condId, bool* condVal, bool* condIsConst) {
  ir::Instruction* cInst = def_use_mgr_->GetDef(condId);
  switch (cInst->opcode()) {
  case SpvOpConstantFalse: {
    *condVal = false;
    *condIsConst = true;
  } break;
  case SpvOpConstantTrue: {
    *condVal = true;
    *condIsConst = true;
  } break;
  case SpvOpLogicalNot: {
    (void)SSAMemGetConstCondition(cInst->GetInOperand(0).words[0],
        condVal, condIsConst);
    if (*condIsConst)
      *condVal = !*condVal;
  } break;
  default: {
    *condIsConst = false;
  } break;
  }
}

void SSAMemPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::vector<ir::Operand> branch_in_operands;
  branch_in_operands.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{labelId}));
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0, branch_in_operands));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void SSAMemPass::SSAMemKillBlk(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    def_use_mgr_->KillInst(ip);
  });
}

bool SSAMemPass::DeadBranchEliminate(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranchConditional)
      continue;
    --ii;
    ir::Instruction* mergeInst = &*ii;
    // GSF TBD Verify that both branches are also sequences of
    // structured control flow
    if (mergeInst->opcode() != SpvOpSelectionMerge)
      continue;
    bool condVal;
    bool condIsConst;
    (void) SSAMemGetConstCondition(
        br->GetInOperand(kSpvBranchCondConditionalId).words[0],
        &condVal,
        &condIsConst);
    if (!condIsConst)
      continue;
    // Replace conditional branch with unconditional branch
    uint32_t trueLabId = br->GetInOperand(kSpvBranchCondTrueLabId).words[0];
    uint32_t falseLabId = br->GetInOperand(kSpvBranchCondFalseLabId).words[0];
    uint32_t mergeLabId = mergeInst->GetInOperand(kSpvSelectionMergeMergeBlockId).words[0];
    uint32_t liveLabId = condVal == true ? trueLabId : falseLabId;
    uint32_t deadLabId = condVal == true ? falseLabId : trueLabId;
    AddBranch(liveLabId, &*bi);
    def_use_mgr_->KillInst(br);
    def_use_mgr_->KillInst(mergeInst);
    // iterate to merge block deleting dead blocks
    std::unordered_set<uint32_t> deadLabIds;
    deadLabIds.insert(deadLabId);
    auto dbi = bi;
    ++dbi;
    uint32_t dLabId = dbi->GetLabelId();
    while (dLabId != mergeLabId) {
      if (deadLabIds.find(dLabId) != deadLabIds.end()) {
        // add successor blocks to dead block set
        dbi->ForEachSucc([&deadLabIds](uint32_t succ) {
          deadLabIds.insert(succ);
        });
        // Add merge block to dead block set in case it has
        // no predecessors.
        uint32_t dMergeLabId = GetMergeBlkId(&*dbi);
        if (dMergeLabId != 0)
          deadLabIds.insert(dMergeLabId);
        // Kill use/def for all instructions and delete block
        SSAMemKillBlk(&*dbi);
        dbi = dbi.Erase();
      }
      else {
        ++dbi;
      }
      dLabId = dbi->GetLabelId();
    }
    // process phi instructions in merge block
    // deadLabIds are now blocks which cannot precede merge block.
    // if eliminated branch is to merge label, add current block to dead blocks.
    if (deadLabId == mergeLabId)
      deadLabIds.insert(bi->GetLabelId());
    dbi->ForEachPhiInst([&deadLabIds, this](ir::Instruction* phiInst) {
      uint32_t phiLabId0 = phiInst->GetInOperand(kSpvPhiLab0Id).words[0];
      bool useFirst = deadLabIds.find(phiLabId0) == deadLabIds.end();
      uint32_t phiValIdx = useFirst ? kSpvPhiVal0Id : kSpvPhiVal1Id;
      uint32_t replId = phiInst->GetInOperand(phiValIdx).words[0];
      uint32_t phiId = phiInst->result_id();
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(phiInst);
    });
    modified = true;
  }
  return modified;
}

bool SSAMemPass::BlockMerge(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ) {
    // Do not merge loop header blocks, at least for now.
    if (IsLoopHeader(&*bi)) {
      ++bi;
      continue;
    }
    // Find block with single successor which has
    // no other predecessors
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranch) {
      ++bi;
      continue;
    }
    uint32_t labId = br->GetInOperand(0).words[0];
    analysis::UseList* uses = def_use_mgr_->GetUses(labId);
    if (uses->size() > 1) {
      ++bi;
      continue;
    }
    // Merge blocks
    def_use_mgr_->KillInst(br);
    auto sbi = bi;
    for (; sbi != func->end(); ++sbi)
      if (sbi->GetLabelId() == labId)
        break;
    assert(sbi != func->end());
    bi->AddInstructions(&*sbi);
    def_use_mgr_->KillInst(sbi->GetLabelInst());
    (void) sbi.Erase();
    // reprocess block
    modified = true;
  }
  return modified;
}

void SSAMemPass::FindCalledFuncs(uint32_t funcId) {
  ir::Function* func = id2function_[funcId];
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpFunctionCall)
        continue;
      uint32_t calledFuncId =
        ii->GetSingleWordInOperand(kSpvFunctionCallFunctionId);
      if (live_func_ids_.find(calledFuncId) != live_func_ids_.end())
        continue;
      called_func_ids_.push(calledFuncId);
    }
  }
}

bool SSAMemPass::DeadFunctionElim() {
  bool modified = false;
  live_func_ids_.clear();
  for (auto& e : module_->entry_points())
    called_func_ids_.push(e.GetSingleWordOperand(kSpvEntryPointFunctionId));
  for (auto& a : module_->annotations()) {
    if (a.opcode() != SpvOpDecorate)
      continue;
    if (a.GetSingleWordInOperand(kSpvDecorationDecoration) != SpvDecorationLinkageAttributes)
      continue;
    if (a.GetSingleWordInOperand(kSpvDecorationLinkageType) != SpvLinkageTypeExport)
      continue;
    uint32_t funcId = a.GetSingleWordInOperand(kSpvDecorationTargetId);
    // Verify that target id maps to a function
    const auto fii = id2function_.find(funcId);
    if (fii == id2function_.end())
      continue;
    called_func_ids_.push(funcId);
  }
  while (!called_func_ids_.empty()) {
    uint32_t funcId = called_func_ids_.front();
    called_func_ids_.pop();
    if (live_func_ids_.find(funcId) != live_func_ids_.end())
      continue;
    live_func_ids_.insert(funcId);
    FindCalledFuncs(funcId);
  }
  auto fni = module_->begin();
  while (fni != module_->end()) {
    if (live_func_ids_.find(fni->GetResultId()) == live_func_ids_.end()) {
      fni = fni.Erase();
      modified = true;
    }
    else {
      ++fni;
    }
  }
  return modified;
}

bool SSAMemPass::SSAMem(ir::Function* func) {
    bool modified = false;
    modified |= LocalAccessChainConvert(func);
    modified |= LocalSingleBlockElim(func);
    modified |= LocalSingleStoreElim(func);
    modified |= InsertExtractElim(func);
    modified |= InsertCycleBreak(func);
    modified |= DeadBranchEliminate(func);
    modified |= BlockMerge(func);

    modified |= LocalSingleBlockElim(func);
    modified |= LocalSingleStoreElim(func);

    modified |= LocalSSARewrite(func);
    modified |= InsertExtractElim(func);
    modified |= FuncDCE(func);

    modified |= UniformAccessChainConvert(func);
    modified |= CommonUniformLoadElimination(func);
    modified |= CommonExtractElimination(func);

    return modified;
}

void SSAMemPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  label2block_.clear();
  for (auto& fn : *module_) {
    id2function_[fn.GetResultId()] = &fn;
    for (auto& blk : fn) {
      uint32_t bid = blk.GetLabelId();
      label2block_[bid] = &blk;
    }
  }

  // Initialize Target Type Caches
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();
};

Pass::Status SSAMemPass::ProcessImpl() {
  bool modified = false;

  // modified |= DeadFunctionElim();

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize next unused Id
  next_id_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    next_id_ = std::max(next_id_, id_def.first);
  }
  ++next_id_;

  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetOperand(kSpvEntryPointFunctionId).words[0]];
    modified = modified || SSAMem(fn);
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

SSAMemPass::SSAMemPass()
    : next_id_(0), module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status SSAMemPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

