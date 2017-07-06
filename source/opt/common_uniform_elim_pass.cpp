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

#include "common_uniform_elim_pass.h"

#include "iterator.h"

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

ir::Instruction* SSAMemPass::GetPtr(ir::Instruction* ip, uint32_t* varId) {
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

bool SSAMemPass::IsUniformVar(uint32_t varId) {
  const ir::Instruction* varInst =
    def_use_mgr_->id_to_defs().find(varId)->second;
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst =
    def_use_mgr_->id_to_defs().find(varTypeId)->second;
  return varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) ==
    SpvStorageClassUniform ||
    varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) ==
    SpvStorageClassUniformConstant;
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

uint32_t SSAMemPass::GetPointeeTypeId(const ir::Instruction* ptrInst) {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const ir::Instruction* ptrTypeInst = def_use_mgr_->GetDef(ptrTypeId);
  return ptrTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeId);
}

void SSAMemPass::GenACLoadRepl(const ir::Instruction* ptrInst,
  std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
  uint32_t& resultId) {

  // Build and append Load
  const uint32_t ldResultId = TakeNextId();
  const uint32_t varId =
    ptrInst->GetSingleWordInOperand(kSpvAccessChainPtrId);
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varPteTypeId = GetPointeeTypeId(varInst);
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
  const uint32_t ptrPteTypeId = GetPointeeTypeId(ptrInst);
  std::vector<ir::Operand> ext_in_opnds;
  ext_in_opnds.push_back(
    ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
      std::initializer_list<uint32_t>{ldResultId}));
  uint32_t iidIdx = 0;
  ptrInst->ForEachInId([&iidIdx, &ext_in_opnds, this](const uint32_t *iid) {
    if (iidIdx > 0) {
      const ir::Instruction* cInst = def_use_mgr_->GetDef(*iid);
      uint32_t val = cInst->GetSingleWordInOperand(kSpvConstantValue);
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

bool CommonUniformEliminate::EliminateCommonUniform(ir::Function* func) {
    bool modified = false;
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

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize next unused Id.
  next_id_ = module->id_bound();
};

Pass::Status SSAMemPass::ProcessImpl() {
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordInOperand(kSpvEntryPointFunctionId);
    modified = EliminateCommonUniform(fn) || modified;
  }
  FinalizeNextId(module_);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

SSAMemPass::SSAMemPass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status SSAMemPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

