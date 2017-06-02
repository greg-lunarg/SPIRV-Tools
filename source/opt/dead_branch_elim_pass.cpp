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

#include "dead_branch_elim_pass.h"

#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;
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

void DeadBranchElimPass::GetConstCondition(
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
    (void)GetConstCondition(cInst->GetSingleWordInOperand(0),
        condVal, condIsConst);
    if (*condIsConst)
      *condVal = !*condVal;
  } break;
  default: {
    *condIsConst = false;
  } break;
  }
}

void DeadBranchElimPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void DeadBranchElimPass::KillBlk(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    def_use_mgr_->KillInst(ip);
  });
}

uint32_t DeadBranchElimPass::GetMergeBlkId(ir::BasicBlock* block_ptr) {
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

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranchConditional)
      continue;
    --ii;
    ir::Instruction* mergeInst = &*ii;
    if (mergeInst->opcode() != SpvOpSelectionMerge)
      continue;
    bool condVal;
    bool condIsConst;
    (void) GetConstCondition(
        br->GetSingleWordInOperand(kSpvBranchCondConditionalId),
        &condVal,
        &condIsConst);
    if (!condIsConst)
      continue;
    // Replace conditional branch with unconditional branch
    uint32_t trueLabId = br->GetSingleWordInOperand(kSpvBranchCondTrueLabId);
    uint32_t falseLabId = br->GetSingleWordInOperand(kSpvBranchCondFalseLabId);
    uint32_t mergeLabId = mergeInst->GetSingleWordInOperand(kSpvSelectionMergeMergeBlockId);
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
    uint32_t dLabId = dbi->id();
    while (dLabId != mergeLabId) {
      if (deadLabIds.find(dLabId) != deadLabIds.end()) {
        // add successor blocks to dead block set
        dbi->ForEachSuccessorLabel([&deadLabIds](uint32_t succ) {
          deadLabIds.insert(succ);
        });
        // Add merge block to dead block set in case it has
        // no predecessors.
        uint32_t dMergeLabId = GetMergeBlkId(&*dbi);
        if (dMergeLabId != 0)
          deadLabIds.insert(dMergeLabId);
        // Kill use/def for all instructions and delete block
        KillBlk(&*dbi);
        dbi = dbi.Erase();
      }
      else {
        ++dbi;
      }
      dLabId = dbi->id();
    }
    // process phi instructions in merge block
    // deadLabIds are now blocks which cannot precede merge block.
    // if eliminated branch is to merge label, add current block to dead blocks.
    if (deadLabId == mergeLabId)
      deadLabIds.insert(bi->id());
    dbi->ForEachPhiInst([&deadLabIds, this](ir::Instruction* phiInst) {
      uint32_t phiLabId0 = phiInst->GetSingleWordInOperand(kSpvPhiLab0Id);
      bool useFirst = deadLabIds.find(phiLabId0) == deadLabIds.end();
      uint32_t phiValIdx = useFirst ? kSpvPhiVal0Id : kSpvPhiVal1Id;
      uint32_t replId = phiInst->GetSingleWordInOperand(phiValIdx);
      uint32_t phiId = phiInst->result_id();
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(phiInst);
    });
    modified = true;
  }
  return modified;
}

void DeadBranchElimPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_)
    id2function_[fn.result_id()] = &fn;

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
};

Pass::Status DeadBranchElimPass::ProcessImpl() {
  bool modified = false;

  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || EliminateDeadBranches(fn);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

DeadBranchElimPass::DeadBranchElimPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status DeadBranchElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

