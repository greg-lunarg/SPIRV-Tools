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

static const int SPV_ENTRY_POINT_FUNCTION_ID = 1;
static const int SPV_STORE_PTR_ID = 0;
static const int SPV_STORE_VAL_ID = 1;
static const int SPV_ACCESS_CHAIN_PTR_ID = 0;
static const int SPV_ACCESS_CHAIN_IDX0_ID = 1;
static const int SPV_TYPE_PTR_STORAGE_CLASS = 0;
static const int SPV_TYPE_PTR_TYPE_ID = 1;
static const int SPV_LOAD_PTR_ID = 0;
static const int SPV_CONSTANT_VALUE = 0;
static const int SPV_EXTRACT_COMPOSITE_ID = 0;
static const int SPV_INSERT_OBJECT_ID = 0;
static const int SPV_INSERT_COMPOSITE_ID = 1;
static const int SPV_BRANCH_TARGET_LAB_ID = 0;
static const int SPV_BRANCH_COND_CONDITIONAL_ID = 0;
static const int SPV_BRANCH_COND_TRUE_LAB_ID = 1;
static const int SPV_BRANCH_COND_FALSE_LAB_ID = 2;
static const int SPV_SEL_MERGE_LAB_ID = 0;
static const int SPV_PHI_VAL0_ID = 0;
static const int SPV_PHI_LAB0_ID = 1;
static const int SPV_PHI_VAL1_ID = 2;
static const int SPV_MERGE_LOOP_BLK_ID = 0;


namespace spvtools {
namespace opt {

bool SSAMemPass::isMathType(const ir::Instruction* typeInst) {
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

bool SSAMemPass::isTargetType(const ir::Instruction* typeInst) {
  if (isMathType(typeInst))
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
    if (!isMathType(compTypeInst)) nonMathComp++;
  });
  return nonMathComp == 0;
}

ir::Instruction* SSAMemPass::GetPtr(ir::Instruction* ip, uint32_t& varId) {
  const uint32_t ptrId = ip->opcode() == SpvOpStore ?
      ip->GetInOperand(SPV_STORE_PTR_ID).words[0] :
      ip->GetInOperand(SPV_LOAD_PTR_ID).words[0];
  ir::Instruction* ptrInst =
    def_use_mgr_->id_to_defs().find(ptrId)->second;
  varId = ptrInst->opcode() == SpvOpAccessChain ?
    ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
    ptrId;
  return ptrInst;
}

bool SSAMemPass::isTargetVar(uint32_t varId) {
  if (seenNonTargetVars.find(varId) != seenNonTargetVars.end())
    return false;
  if (seenTargetVars.find(varId) != seenTargetVars.end())
    return true;
  const ir::Instruction* varInst =
    def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
    def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetInOperand(SPV_TYPE_PTR_STORAGE_CLASS).words[0] !=
    SpvStorageClassFunction) {
    seenNonTargetVars.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
    varTypeInst->GetInOperand(SPV_TYPE_PTR_TYPE_ID).words[0];
  ir::Instruction* varPteTypeInst =
    def_use_mgr_->id_to_defs().find(varPteTypeId)->second;
  if (!isTargetType(varPteTypeInst)) {
    seenNonTargetVars.insert(varId);
    return false;
  }
  seenTargetVars.insert(varId);
  return true;
}

void SSAMemPass::SSAMemAnalyze(ir::Function* func) {
  ssaVars.clear();
  ssaComps.clear();
  ssaCompVars.clear();
  nonSsaVars.clear();
  storeIdx.clear();
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++, instIdx++) {
      if (ii->opcode() != SpvOpStore)
        continue;

      // Verify store variable is target type
      uint32_t varId;
      ir::Instruction* ptrInst = GetPtr(&*ii, varId);
      if (!isTargetVar(varId)) {
        nonSsaVars.insert(varId);
        continue;
      }

      // Verify store variable/component is not yet assigned.
      // If whole variable is stored, there should not be any
      // component stores and vice-versa.
      if (nonSsaVars.find(varId) != nonSsaVars.end())
        continue;
      if (ssaVars.find(varId) != ssaVars.end()) {
        nonSsaVars.insert(varId);
        continue;
      }
      if (ptrInst->opcode() == SpvOpAccessChain) {
        if (ptrInst->NumInOperands() != 2) {
          nonSsaVars.insert(varId);
          continue;
        }
        const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
        if (ssaComps.find(std::make_pair(varId, idxId)) != ssaComps.end()) {
          nonSsaVars.insert(varId);
          continue;
        }
        ssaCompVars.insert(varId);
        ssaComps[std::make_pair(varId, idxId)] = &*ii;
        storeIdx[&*ii] = instIdx;
      }
      else {
        if (ssaCompVars.find(varId) != ssaCompVars.end()) {
          nonSsaVars.insert(varId);
          continue;
        }
        ssaVars[varId] = &*ii;
        storeIdx[&*ii] = instIdx;
      }
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

bool SSAMemPass::SSAMemProcess(ir::Function* func) {
  bool modified = false;
  uint32_t instIdx = 0;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++, instIdx++) {
      if (ii->opcode() != SpvOpLoad)
        continue;
      const uint32_t ptrId = ii->GetInOperand(SPV_LOAD_PTR_ID).words[0];
      ir::Instruction* ptrInst =
          def_use_mgr_->id_to_defs().find(ptrId)->second;
      const uint32_t varId = ptrInst->opcode() == SpvOpAccessChain ?
          ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0] :
          ptrId;
      const ir::Instruction* varInst =
          def_use_mgr_->id_to_defs().find(varId)->second;
      assert(varInst->opcode() == SpvOpVariable);
      if (nonSsaVars.find(varId) != nonSsaVars.end())
        continue;
      uint32_t replId;
      uint32_t sIdx;
      if (ptrInst->opcode() == SpvOpAccessChain) {
        if (ptrInst->NumInOperands() != 2)
          continue;
        // process component load
        const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
        const auto csi = ssaComps.find(std::make_pair(varId, idxId));
        if (csi != ssaComps.end()) {
          replId = csi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          sIdx = storeIdx[csi->second];
        }
        else {
          // See if the whole variable stored with a load of an SSA var.
          // If so, look for a component store into the load variable
          // and use the value Id that was stored.
          const auto vsi = ssaVars.find(varId);
          if (vsi == ssaVars.end())
            continue;
          const uint32_t valId =
              vsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          const ir::Instruction* valInst =
              def_use_mgr_->id_to_defs().find(valId)->second;
          if (valInst->opcode() != SpvOpLoad)
            continue;
          const uint32_t loadVarId =
              valInst->GetInOperand(SPV_LOAD_PTR_ID).words[0];
          if (nonSsaVars.find(loadVarId) != nonSsaVars.end())
            continue;
          const auto lvcsi = ssaComps.find(std::make_pair(loadVarId, idxId));
          if (lvcsi == ssaComps.end())
            continue;
          replId = lvcsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          sIdx = storeIdx[lvcsi->second];
        }
      }
      else {
        // process whole variable load
        const auto vsi = ssaVars.find(varId);
        // if variable is not defined with whole variable store,
        // skip this load
        if (vsi == ssaVars.end())
          continue;
        replId = vsi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
        sIdx = storeIdx[vsi->second];
      }
      // store must dominate load
      if (instIdx < sIdx)
        continue;
      // replace all instances of the load's id with the SSA value's id
      ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
      modified = true;
    }
  }
  return modified;
}

bool SSAMemPass::isLiveVar(uint32_t varId) {
  // non-function scope vars are live
  const ir::Instruction* varInst =
      def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
      def_use_mgr_->id_to_defs().find(varTypeId)->second;
  if (varTypeInst->GetInOperand(SPV_TYPE_PTR_STORAGE_CLASS).words[0] !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
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

bool SSAMemPass::isLiveStore(ir::Instruction* storeInst) {
  // get store's variable
  uint32_t varId;
  (void) GetPtr(storeInst, varId);
  return isLiveVar(varId);
}

void SSAMemPass::DCEInst(ir::Instruction* inst) {
  std::queue<ir::Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    ir::Instruction* di = deadInsts.front();
    std::queue<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) {
      ids.push(*iid);
    });
    uint32_t varId = 0;
    if (di->opcode() == SpvOpLoad)
      (void) GetPtr(di, varId);
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
    if (varId != 0 && !isLiveVar(varId)) {
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

bool SSAMemPass::SSAMemDCE() {
  bool modified = false;
  for (auto v : ssaVars) {
    // check that it hasn't already been DCE'd
    if (v.second->opcode() != SpvOpStore)
      continue;
    if (nonSsaVars.find(v.first) != nonSsaVars.end())
      continue;
    if (!isLiveStore(v.second)) {
      DCEInst(v.second);
      modified = true;
    }
  }
  for (auto c : ssaComps) {
    // check that it hasn't already been DCE'd
    if (c.second->opcode() != SpvOpStore)
      continue;
    if (nonSsaVars.find(c.first.first) != nonSsaVars.end())
      continue;
    if (!isLiveStore(c.second)) {
      DCEInst(c.second);
      modified = true;
    }
  }
  return modified;
}

void SSAMemPass::SBEraseComps(uint32_t varId) {
  for (auto ci = sbCompStores.begin(); ci != sbCompStores.end();)
    if (ci->first.first == varId)
      ci = sbCompStores.erase(ci);
    else
      ci++;
  for (auto ci = sbPinnedComps.begin(); ci != sbPinnedComps.end();)
    if (ci->first == varId)
      ci = sbPinnedComps.erase(ci);
    else
      ci++;
}

bool SSAMemPass::SSAMemSingleBlock(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    sbVarStores.clear();
    sbVarLoads.clear();
    sbCompStores.clear();
    sbPinnedVars.clear();
    sbPinnedComps.clear();
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetVar(varId))
          continue;
        // Register the store
        if (ptrInst->opcode() == SpvOpVariable) {
          // if not pinned, look for WAW
          if (sbPinnedVars.find(varId) == sbPinnedVars.end()) {
            auto si = sbVarStores.find(varId);
            if (si != sbVarStores.end()) {
              def_use_mgr_->KillInst(si->second);
            }
          }
          sbVarStores[varId] = &*ii;
          SBEraseComps(varId);
        }
        else {
          assert(ptrInst->opcode() == SpvOpAccessChain);
          const uint32_t idxId =
              ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
          if (ptrInst->NumInOperands() == 2) {
            // if not pinned, look for WAW
            if (sbPinnedComps.find(std::make_pair(varId, idxId)) == sbPinnedComps.end()) {
              auto si = sbCompStores.find(std::make_pair(varId, idxId));
              if (si != sbCompStores.end()) {
                DCEInst(si->second);
              }
            }
            sbCompStores[std::make_pair(varId, idxId)] = &*ii;
          }
          else
            sbCompStores.erase(std::make_pair(varId, idxId));
          sbPinnedComps.erase(std::make_pair(varId, idxId));
          sbVarStores.erase(varId);
        }
        sbPinnedVars.erase(varId);
        sbVarLoads.erase(varId);
      } break;
      case SpvOpLoad: {
        // Verify store variable is target type
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetVar(varId))
          continue;
        // Look for previous store or load
        uint32_t replId = 0;
        if (ptrInst->opcode() == SpvOpVariable) {
          auto si = sbVarStores.find(varId);
          if (si != sbVarStores.end()) {
            replId = si->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          }
          else {
            auto li = sbVarLoads.find(varId);
            if (li != sbVarLoads.end()) {
              replId = li->second->result_id();
            }
          }
        }
        else if (ptrInst->NumInOperands() == 2) {
          assert(ptrInst->opcode() == SpvOpAccessChain);
          const uint32_t idxId =
            ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
          auto ci = sbCompStores.find(std::make_pair(varId, idxId));
          if (ci != sbCompStores.end()) {
            replId = ci->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
          }
          else {
            // if live store to whole variable of load, look for component
            // store to the loaded variable
            auto si = sbVarStores.find(varId);
            if (si != sbVarStores.end()) {
              const uint32_t valId =
                si->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
              ir::Instruction* valInst =
                def_use_mgr_->id_to_defs().find(valId)->second;
              if (valInst->opcode() == SpvOpLoad) {
                uint32_t loadVarId =
                  valInst->GetInOperand(SPV_LOAD_PTR_ID).words[0];
                auto lvi = sbCompStores.find(std::make_pair(loadVarId, idxId));
                if (lvi != sbCompStores.end()) {
                  replId = lvi->second->GetInOperand(SPV_STORE_VAL_ID).words[0];
                }
              }
            }
          }
        }
        if (replId != 0) {
          // replace load's result id and delete load
          ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
          modified = true;
        }
        else {
          if (ptrInst->opcode() == SpvOpVariable) {
            sbVarLoads[varId] = &*ii;  // register load
            sbPinnedVars.insert(varId);
            for (auto ci = sbCompStores.begin(); ci != sbCompStores.end(); ci++)
              if (ci->first.first == varId)
                sbPinnedComps.insert(std::make_pair(varId, ci->first.second));
          }
          else {
            const uint32_t idxId =
              ptrInst->GetInOperand(SPV_ACCESS_CHAIN_IDX0_ID).words[0];
            if (sbCompStores.find(std::make_pair(varId, idxId)) != sbCompStores.end())
              sbPinnedComps.insert(std::make_pair(varId, idxId));
            else
              sbPinnedVars.insert(varId);
          }
        }
      } break;
      default:
        break;
      }
    }
    // Go back and delete useless stores in block
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      if (ii->opcode() != SpvOpStore)
        continue;
      if (isLiveStore(&*ii))
        continue;
      DCEInst(&*ii);
    }
  }
  return modified;
}

bool SSAMemPass::SSAMemDCEFunc(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        (void)GetPtr(&*ii, varId);
        if (isLiveVar(varId))
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
  return ptrTypeInst->GetInOperand(SPV_TYPE_PTR_TYPE_ID).words[0];
}

void SSAMemPass::GenACLoadRepl(const ir::Instruction* ptrInst,
  std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
  uint32_t& resultId) {

  // Build and append Load
  const uint32_t ldResultId = getNextId();
  const uint32_t varId =
    ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0];
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
  const uint32_t extResultId = getNextId();
  const uint32_t ptrPteTypeId = GetPteTypeId(ptrInst);
  std::vector<ir::Operand> ext_in_opnds;
  ext_in_opnds.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{ldResultId}));
  uint32_t iidIdx = 0;
  ptrInst->ForEachInId([&iidIdx, &ext_in_opnds, this](const uint32_t *iid) {
    if (iidIdx > 0) {
      const ir::Instruction* cInst = def_use_mgr_->GetDef(*iid);
      uint32_t val = cInst->GetInOperand(SPV_CONSTANT_VALUE).words[0];
      ext_in_opnds.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          std::initializer_list<uint32_t>{val}));
    }
    iidIdx++;
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
  const uint32_t ldResultId = getNextId();
  const uint32_t varId =
    ptrInst->GetInOperand(SPV_ACCESS_CHAIN_PTR_ID).words[0];
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
  const uint32_t insResultId = getNextId();
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
      uint32_t val = cInst->GetInOperand(SPV_CONSTANT_VALUE).words[0];
      ins_in_opnds.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          std::initializer_list<uint32_t>{val}));
    }
    iidIdx++;
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

bool SSAMemPass::SSAMemAccessChainRemoval(ir::Function* func) {
  // rule out variables accessed with non-constant indices
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore:
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!isTargetVar(varId))
          break;
        uint32_t inIdx = 0;
        uint32_t nonConstCnt = 0;
        ptrInst->ForEachInId([&inIdx,&nonConstCnt,this](uint32_t* tid) {
          if (inIdx > 0) {
            ir::Instruction* opInst = def_use_mgr_->GetDef(*tid);
            if (opInst->opcode() != SpvOpConstant) nonConstCnt++;
          }
          inIdx++;
        });
        if (nonConstCnt > 0) {
          seenNonTargetVars.insert(varId);
          seenTargetVars.erase(varId);
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
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!isTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        if (ii->opcode() == SpvOpLoad) {
          uint32_t replId;
          GenACLoadRepl(ptrInst, newInsts, replId);
          ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
          ii++;
          ii = ii.MoveBefore(newInsts);
          ii++;
        }
        else {
          uint32_t valId = ii->GetInOperand(SPV_STORE_VAL_ID).words[0];
          GenACStoreRepl(ptrInst, valId, newInsts);
          def_use_mgr_->KillInst(&*ii);
          DeleteIfUseless(ptrInst);
          ii++;
          ii = ii.MoveBefore(newInsts);
          ii++;
          ii++;
        }
        modified = true;
      } break;
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (ptrInst->opcode() != SpvOpAccessChain)
          break;
        if (!isTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        if (ii->opcode() == SpvOpLoad) {
          uint32_t replId;
          GenACLoadRepl(ptrInst, newInsts, replId);
          ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
          ii++;
          ii = ii.MoveBefore(newInsts);
          ii++;
        }
        else {
          uint32_t valId = ii->GetInOperand(SPV_STORE_VAL_ID).words[0];
          GenACStoreRepl(ptrInst, valId, newInsts);
          def_use_mgr_->KillInst(&*ii);
          DeleteIfUseless(ptrInst);
          ii++;
          ii = ii.MoveBefore(newInsts);
          ii++;
          ii++;
        }
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  return modified;
}

bool SSAMemPass::IsStructured(ir::Function* func) {
  // TODO(greg-lunarg)
  return true;
}

bool SSAMemPass::IsLoopHeader(ir::BasicBlock* block_ptr) {
  auto inst = block_ptr->begin();
  inst++;
  return inst->opcode() == SpvOpLoopMerge;
}

void SSAMemPass::SSABlockInitSinglePred(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  uint32_t predLabel = label2preds_[label].front();
  label2SSA_[label] = label2SSA_[predLabel];
}

bool SSAMemPass::HasStore(uint32_t var_id, uint32_t first_label,
  uint32_t last_label) {
  // TODO(greg-lunarg)
  return true;
}

bool SSAMemPass::HasLoad(uint32_t var_id, uint32_t label) {
  // TODO(greg-lunarg)
  return true;
}

void SSAMemPass::SSABlockInitLoopHeader(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  auto mergeInst = block_ptr->begin();
  mergeInst++;
  uint32_t mergeLabel = mergeInst->GetInOperand(SPV_MERGE_LOOP_BLK_ID).words[0];
  uint32_t pred0Label = 0;
  uint32_t backLabel = 0;
  for (uint32_t predLabel : label2preds_[label])
    if (visitedBlocks.find(predLabel) == visitedBlocks.end())
      backLabel = predLabel;
    else if (pred0Label == 0)
      pred0Label = predLabel;
  assert(pred0Label != 0);
  assert(backLabel != 0);
  for (auto var_val : label2SSA_[pred0Label]) {
    uint32_t varId = var_val.first;
    if (!HasLoad(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool differs = false;
    for (uint32_t predLabel : label2preds_[label]) {
      if (predLabel == pred0Label)
        continue;
      // Skip back edge predecessor.
      if (predLabel == backLabel)
        continue;
      const auto var_val_itr = label2SSA_[predLabel].find(varId);
      // Missing values do not cause difference
      if (var_val_itr == label2SSA_[predLabel].end())
        continue;
      if (var_val_itr->second != val0Id) {
        differs = true;
        break;
      }
    }
    // If variable is stored in loop, back edge predecessor
    // introduces difference.
    if (!differs && HasStore(varId, label, mergeLabel))
      differs = true;
    // If val is the same for all predecessors, enter it in map
    if (!differs) {
      label2SSA_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map. For back edge predecessor,
    // use the variable id. We will fix this after visiting back
    // edge predecessor.
    std::vector<ir::Operand> phi_in_operands;
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2SSA_[predLabel].find(varId);
      const uint32_t valId = (predLabel == backLabel) ? varId :
          (var_val_itr != label2SSA_[predLabel].end() ? 
            var_val_itr->second : val0Id);
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t resultId = getNextId();
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpBranch, typeId, resultId, phi_in_operands));
    block_ptr->begin().InsertBefore(std::move(newPhi));
    label2SSA_[label].insert({ varId, resultId });
  }
}

void SSAMemPass::SSABlockInitSelectMerge(ir::BasicBlock* block_ptr) {
  uint32_t label = block_ptr->GetLabelId();
  uint32_t pred0Label = label2preds_[label].front();
  for (auto var_val : label2SSA_[pred0Label]) {
    uint32_t varId = var_val.first;
    if (!HasLoad(varId, label))
      continue;
    uint32_t val0Id = var_val.second;
    bool differs = false;
    for (uint32_t predLabel : label2preds_[label]) {
      if (predLabel == pred0Label)
        continue;
      const auto var_val_itr = label2SSA_[predLabel].find(varId);
      // Missing values do not cause a difference
      if (var_val_itr == label2SSA_[predLabel].end())
        continue;
      if (var_val_itr->second != val0Id) {
        differs = true;
        break;
      }
    }
    // If val is the same for all predecessors, enter it in map
    if (!differs) {
      label2SSA_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and 
    // add its result id to the map
    std::vector<ir::Operand> phi_in_operands;
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2SSA_[predLabel].find(varId);
      // TODO(greg-lunarg) It is possible although rare that a live
      // phi function could not have a value defined from one (or more)
      // predecessors. The proper way to handle this to insert an undef
      // into the predecessor block in question and use that as the phi
      // operand. In the meantime, we will just use one of the other valid
      // values. If the phi lives, the validator will likely complain the
      // the definition of the value does not dominate the predecessor.
      uint32_t valId = (var_val_itr != label2SSA_[predLabel].end()) ?
          var_val_itr->second : val0Id;
      phi_in_operands.push_back(
          ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{valId}));
      phi_in_operands.push_back(
        ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{predLabel}));
    }
    uint32_t resultId = getNextId();
    uint32_t typeId = GetPteTypeId(def_use_mgr_->GetDef(varId));
    std::unique_ptr<ir::Instruction> newPhi(
      new ir::Instruction(SpvOpBranch, typeId, resultId, phi_in_operands));
    block_ptr->begin().InsertBefore(std::move(newPhi));
    label2SSA_[label].insert({varId, resultId});
  }
}

void SSAMemPass::SSABlockInit(ir::BasicBlock* block_ptr) {
  size_t numPreds = label2preds_[block_ptr->GetLabelId()].size();
  if (numPreds == 0)
    return;
  if (numPreds == 1)
    SSABlockInitSinglePred(block_ptr);
  else if (IsLoopHeader(block_ptr))
    SSABlockInitLoopHeader(block_ptr);
  else
    SSABlockInitSelectMerge(block_ptr);
}

bool SSAMemPass::SSAMemSSARewrite(ir::Function* func) {
  if (!IsStructured(func))
    return false;
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    SSABlockInit(&*bi);
    uint32_t label = bi->GetLabelId();
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        uint32_t valId = ii->GetInOperand(SPV_STORE_VAL_ID).words[0];
        label2SSA_[label][varId] = valId;
      } break;
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (!isTargetVar(varId))
          break;
        assert(ptrInst->opcode() != SpvOpAccessChain);
        modified = true;
      } break;
      default: {
      } break;
      }
    }
    visitedBlocks.insert(label);
  }
  return modified;
}


bool SSAMemPass::SSAMemExtInsMatch(ir::Instruction* extInst,
    ir::Instruction* insInst) {
  if (extInst->NumInOperands() != insInst->NumInOperands() - 1)
    return false;
  uint32_t numIdx = extInst->NumInOperands() - 1;
  for (uint32_t i = 0; i < numIdx; i++)
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
  for (uint32_t i = 0; i < numIdx; i++)
    if (extInst->GetInOperand(i + 1).words[0] !=
        insInst->GetInOperand(i + 2).words[0])
      return false;
  return true;
}

bool SSAMemPass::SSAMemEliminateExtracts(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpCompositeExtract: {
        uint32_t cid = ii->GetInOperand(SPV_EXTRACT_COMPOSITE_ID).words[0];
        ir::Instruction* cinst = def_use_mgr_->GetDef(cid);
        uint32_t replId = 0;
        while (cinst->opcode() == SpvOpCompositeInsert) {
          if (SSAMemExtInsConflict(&*ii, cinst))
            break;
          if (SSAMemExtInsMatch(&*ii, cinst)) {
            replId = cinst->GetInOperand(SPV_INSERT_OBJECT_ID).words[0];
            break;
          }
          cid = cinst->GetInOperand(SPV_INSERT_COMPOSITE_ID).words[0];
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

bool SSAMemPass::SSAMemBreakLSCycle(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ii++) {
      switch (ii->opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, varId);
        if (ptrInst->opcode() != SpvOpVariable)
          break;
        if (!isTargetVar(varId))
          break;
        analysis::UseList* vuses = def_use_mgr_->GetUses(varId);
        if (vuses->size() > 2)
          break;
        uint32_t valId = ii->GetInOperand(SPV_STORE_VAL_ID).words[0];
        ir::Instruction* vinst = def_use_mgr_->GetDef(valId);
        uint32_t intermediate = false;
        while (vinst->opcode() == SpvOpCompositeInsert) {
          if (intermediate) {
            const uint32_t resId = vinst->result_id();
            analysis::UseList* uses = def_use_mgr_->GetUses(resId);
            if (uses->size() > 1)
              break;
          }
          valId = vinst->GetInOperand(SPV_INSERT_COMPOSITE_ID).words[0];
          vinst = def_use_mgr_->GetDef(valId);
          intermediate = true;
        }
        if (vinst->opcode() != SpvOpLoad)
          break;
        uint32_t lvarId = vinst->GetInOperand(SPV_LOAD_PTR_ID).words[0];
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
    for (auto bi = func->begin(); bi != func->end(); bi++) {
      for (auto ii = bi->begin(); ii != bi->end(); ii++) {
        switch (ii->opcode()) {
        case SpvOpLoad: {
          uint32_t varId;
          ir::Instruction* ptrInst = GetPtr(&*ii, varId);
          if (ptrInst->opcode() != SpvOpVariable)
            break;
          if (!isTargetVar(varId))
            break;
          analysis::UseList* vuses = def_use_mgr_->GetUses(varId);
          if (vuses->size() == 1) {
            uint32_t undefId = getNextId();
            uint32_t typeId = ii->type_id();
            std::vector<ir::Operand> in_operands;
            in_operands.clear();
            std::unique_ptr<ir::Instruction> undef_inst(
              new ir::Instruction(SpvOpUndef, typeId, undefId, in_operands));
            def_use_mgr_->AnalyzeInstDefUse(&*undef_inst);
            ii = ii.InsertBefore(std::move(undef_inst));
            ii++; // back to load
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

bool SSAMemPass::SSAMemDeadBranchEliminate(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    auto ii = bi->end();
    ii--;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranchConditional)
      continue;
    ii--;
    ir::Instruction* mergeInst = &*ii;
    // GSF TBD Verify that both branches are also sequences of
    // structured control flow
    if (mergeInst->opcode() != SpvOpSelectionMerge)
      continue;
    bool condVal;
    bool condIsConst;
    (void) SSAMemGetConstCondition(
        br->GetInOperand(SPV_BRANCH_COND_CONDITIONAL_ID).words[0],
        &condVal,
        &condIsConst);
    if (!condIsConst)
      continue;
    // Replace conditional branch with unconditional branch
    uint32_t trueLabId = br->GetInOperand(SPV_BRANCH_COND_TRUE_LAB_ID).words[0];
    uint32_t falseLabId = br->GetInOperand(SPV_BRANCH_COND_FALSE_LAB_ID).words[0];
    uint32_t mergeLabId = mergeInst->GetInOperand(SPV_SEL_MERGE_LAB_ID).words[0];
    uint32_t liveLabId = condVal == true ? trueLabId : falseLabId;
    uint32_t deadLabId = condVal == true ? falseLabId : trueLabId;
    AddBranch(liveLabId, &*bi);
    def_use_mgr_->KillInst(br);
    def_use_mgr_->KillInst(mergeInst);
    // iterate to merge block deleting dead blocks
    std::unordered_set<uint32_t> deadLabIds;
    deadLabIds.insert(deadLabId);
    auto dbi = bi;
    dbi++;
    uint32_t dLabId = dbi->GetLabelId();
    while (dLabId != mergeLabId) {
      if (deadLabIds.find(dLabId) != deadLabIds.end()) {
        // add successor blocks to dead block set
        dbi->ForEachSucc([&deadLabIds](uint32_t succ) {
          deadLabIds.insert(succ);
        });
        // Kill use/def for all instructions and delete block
        SSAMemKillBlk(&*dbi);
        dbi = dbi.Erase();
      }
      else
        dbi++;
      dLabId = dbi->GetLabelId();
    }
    // process phi instructions in merge block
    // deadLabIds are now blocks which cannot precede merge block.
    // if eliminated branch is to merge label, add current block to dead blocks.
    if (deadLabId == mergeLabId)
      deadLabIds.insert(bi->GetLabelId());
    dbi->ForEachPhiInst([&deadLabIds, this](ir::Instruction* phiInst) {
      uint32_t phiLabId0 = phiInst->GetInOperand(SPV_PHI_LAB0_ID).words[0];
      bool useFirst = deadLabIds.find(phiLabId0) == deadLabIds.end();
      uint32_t phiValIdx = useFirst ? SPV_PHI_VAL0_ID : SPV_PHI_VAL1_ID;
      uint32_t replId = phiInst->GetInOperand(phiValIdx).words[0];
      uint32_t phiId = phiInst->result_id();
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(phiInst);
    });
    modified = true;
  }
  return modified;
}

bool SSAMemPass::SSAMemBlockMerge(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ) {
    // Find block with single successor which has
    // no other predecessors
    auto ii = bi->end();
    ii--;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranch) {
      bi++;
      continue;
    }
    uint32_t labId = br->GetInOperand(0).words[0];
    analysis::UseList* uses = def_use_mgr_->GetUses(labId);
    if (uses->size() > 1) {
      bi++;
      continue;
    }
    // Merge blocks
    def_use_mgr_->KillInst(br);
    auto sbi = bi;
    for (; sbi != func->end(); sbi++)
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

bool SSAMemPass::SSAMem(ir::Function* func) {
    bool modified = false;
    modified |= SSAMemAccessChainRemoval(func);
    modified |= SSAMemSingleBlock(func);
    SSAMemAnalyze(func);
    modified |= SSAMemProcess(func);
    modified |= SSAMemDCE();
    modified |= SSAMemEliminateExtracts(func);
    modified |= SSAMemBreakLSCycle(func);
    modified |= SSAMemDCEFunc(func);
    modified |= SSAMemDeadBranchEliminate(func);
    modified |= SSAMemBlockMerge(func);

    modified |= SSAMemSingleBlock(func);
    SSAMemAnalyze(func);
    modified |= SSAMemProcess(func);
    modified |= SSAMemDCE();

    return modified;
}

void SSAMemPass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id
  nextId_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    nextId_ = std::max(nextId_, id_def.first);
  }
  nextId_++;

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  label2block_.clear();
  for (auto& fn : *module_) {
    id2function_[fn.GetResultId()] = &fn;
    for (auto& blk : fn) {
      uint32_t bid = blk.GetLabelId();
      label2block_[bid] = &blk;
      blk.ForEachSucc([&bid,this](uint32_t sbid) {
        label2preds_[sbid].push_back(bid);
      });
    }
  }

  // Initialize Target Type Caches
  seenTargetVars.clear();
  seenNonTargetVars.clear();
};

Pass::Status SSAMemPass::ProcessImpl() {
  // do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetOperand(SPV_ENTRY_POINT_FUNCTION_ID).words[0]];
    modified = modified || SSAMem(fn);
  }

  finalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

SSAMemPass::SSAMemPass()
    : nextId_(0), module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status SSAMemPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

