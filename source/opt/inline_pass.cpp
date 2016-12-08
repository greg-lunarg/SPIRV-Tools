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

#include "inline_pass.h"
#include "iterator.h"

#include <unordered_map>

#define SPV_ENTRY_POINT_FUNCTION_ID 1
#define SPV_FUNCTION_CALL_FUNCTION_ID 2
#define SPV_FUNCTION_CALL_ARGUMENT_ID 3
#define SPV_FUNCTION_PARAMETER_RESULT_ID 1
#define SPV_STORE_OBJECT_ID 1
#define SPV_RETURN_VALUE_ID 0
#define SPV_TYPE_POINTER_STORAGE_CLASS 1
#define SPV_TYPE_POINTER_TYPE_ID 2

namespace spvtools {
namespace opt {

// Generate callee code into newBlocks to be inlined for the function call at
// call_ii. Also add new function variables into caller func

void InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<ir::BasicBlock>>& newBlocks,
    std::vector<std::unique_ptr<ir::Instruction>>& newVars,
    ir::UptrVectorIterator<ir::Instruction> call_ii,
    ir::UptrVectorIterator<ir::BasicBlock> call_bi) {
  // Map from callee id to caller id
  std::unordered_map<uint32_t, uint32_t> callee2caller;

  uint32_t calleeId =
      call_ii->GetOperand(SPV_FUNCTION_CALL_FUNCTION_ID).words[0];
  ir::Function* calleeFn = id2function_[calleeId];

  // Map parameters to actual arguments
  int i = 0;
  calleeFn->ForEachParam(
      [&call_ii, &i, &callee2caller](const ir::Instruction* cpi) {
        auto pid = cpi->GetOperand(SPV_FUNCTION_PARAMETER_RESULT_ID).words[0];
        callee2caller[pid] =
            call_ii->GetOperand(SPV_FUNCTION_CALL_ARGUMENT_ID + i).words[0];
        i++;
      });

  // Define caller local variables for all callee variables and create map to
  // them
  auto cbi = calleeFn->begin();
  auto cvi = cbi->begin();
  while (cvi->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(*cvi));
    uint32_t newId = getNextId();
    var_inst->SetResultId(newId);
    callee2caller[cvi->result_id()] = newId;
    newVars.push_back(std::move(var_inst));
    cvi++;
  }

  // Create return var if needed
  uint32_t calleeTypeId = calleeFn->GetTypeId();
  ir::Instruction* calleeType =
      def_use_mgr_->id_to_defs().find(calleeTypeId)->second;
  uint32_t returnVarId = 0;
  if (calleeType->opcode() != SpvOpTypeVoid) {
    // find or create ptr to callee return type
    ir::Module::inst_iterator inst_iter = module_->types_values_begin();
    for (; inst_iter != module_->types_values_end(); ++inst_iter) {
      ir::Instruction* type_inst = &*inst_iter;
      if (type_inst->opcode() == SpvOpTypePointer &&
          type_inst->GetOperand(SPV_TYPE_POINTER_TYPE_ID).words[0] ==
              calleeTypeId &&
          type_inst->GetOperand(SPV_TYPE_POINTER_STORAGE_CLASS).words[0] ==
              SpvStorageClassFunction)
        break;
    }
    uint32_t returnVarTypeId;
    if (inst_iter != module_->types_values_end()) {
      returnVarTypeId = inst_iter->result_id();
    } else {
      // add pointer type to module
      returnVarTypeId = getNextId();
      std::vector<ir::Operand> in_operands;
      in_operands.emplace_back(
          spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
          std::initializer_list<uint32_t>{uint32_t(SpvStorageClassFunction)});
      in_operands.emplace_back(
          spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          std::initializer_list<uint32_t>{uint32_t(calleeTypeId)});
      std::unique_ptr<ir::Instruction> type_inst(new ir::Instruction(
          SpvOpTypePointer, 0, returnVarTypeId, in_operands));
      module_->AddType(std::move(type_inst));
    }
    returnVarId = getNextId();
    std::vector<ir::Operand> in_operands;
    in_operands.emplace_back(
        spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
        std::initializer_list<uint32_t>{uint32_t(SpvStorageClassFunction)});
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(
        SpvOpVariable, returnVarTypeId, returnVarId, in_operands));
    newVars.push_back(std::move(var_inst));
  }

  // Clone and map callee code
  bool prevInstWasReturn = false;
  uint32_t returnLabelId = 0;
  std::unique_ptr<ir::BasicBlock> bp;
  calleeFn->ForEachInst([&newBlocks, &callee2caller, &call_bi, &call_ii, &bp,
                         &prevInstWasReturn, &returnLabelId, &returnVarId, 
                         &calleeTypeId, this](const ir::Instruction* cpi) {
    switch (cpi->opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
      case SpvOpVariable:
        // already processed
        break;
      case SpvOpLabel: {
        // if previous instruction was early return, insert branch instruction
        // to return block
        if (prevInstWasReturn) {
          if (returnLabelId == 0) returnLabelId = this->getNextId();
          std::vector<ir::Operand> branch_in_operands;
          branch_in_operands.push_back(
              ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                          std::initializer_list<uint32_t>{returnLabelId}));
          std::unique_ptr<ir::Instruction> newBranch(
              new ir::Instruction(SpvOpBranch, 0, 0, branch_in_operands));
          bp->AddInstruction(std::move(newBranch));
          prevInstWasReturn = false;
        }
        // finish current block (if it exists) and create next block
        uint32_t labelId;
        bool firstBlock = false;
        if (bp != nullptr) {
          newBlocks.push_back(std::move(bp));
          // if result id is already mapped, use it, otherwise get a new one.
          auto rid = cpi->result_id();
          auto s = callee2caller.find(rid);
          labelId = (s != callee2caller.end()) ? s->second : this->getNextId();
        } else {
          // first block needs to use label of original block
          // but map callee label in case of phi reference
          labelId = call_bi->GetLabelId();
          callee2caller[cpi->result_id()] = labelId;
          firstBlock = true;
        }
        const std::vector<ir::Operand> label_in_operands;
        std::unique_ptr<ir::Instruction> newLabel(
            new ir::Instruction(SpvOpLabel, 0, labelId, label_in_operands));
        bp.reset(new ir::BasicBlock(std::move(newLabel)));
        if (firstBlock) {
          // Copy contents of original caller block up to call instruction
          for (auto cii = call_bi->begin(); cii != call_ii; cii++) {
            std::unique_ptr<ir::Instruction> spv_inst(
                new ir::Instruction(*cii));
            bp->AddInstruction(std::move(spv_inst));
          }
        }
      } break;
      case SpvOpReturnValue: {
        // store return value to return variable
        assert(returnVarId != 0);
        auto valId = cpi->GetInOperand(SPV_RETURN_VALUE_ID).words[0];
        auto s = callee2caller.find(valId);
        if (s != callee2caller.end()) {
          valId = s->second;
        }
        std::vector<ir::Operand> store_in_operands;
        store_in_operands.push_back(
            ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                        std::initializer_list<uint32_t>{returnVarId}));
        store_in_operands.push_back(
            ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                        std::initializer_list<uint32_t>{valId}));
        std::unique_ptr<ir::Instruction> newBranch(
            new ir::Instruction(SpvOpStore, 0, 0, store_in_operands));
        bp->AddInstruction(std::move(newBranch));

        // Remember we saw a return; if followed by a label, will need to insert
        // branch
        prevInstWasReturn = true;
      } break;
      case SpvOpReturn: {
        // Remember we saw a return; if followed by a label, will need to insert
        // branch
        prevInstWasReturn = true;
      } break;
      case SpvOpFunctionEnd: {
        // if there was an early return, create return label/block
        // if previous instruction was return, insert branch instruction
        // to return block
        if (returnLabelId != 0) {
          if (prevInstWasReturn) {
            std::vector<ir::Operand> branch_in_operands;
            branch_in_operands.push_back(
                ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                            std::initializer_list<uint32_t>{returnLabelId}));
            std::unique_ptr<ir::Instruction> newBranch(
                new ir::Instruction(SpvOpBranch, 0, 0, branch_in_operands));
            bp->AddInstruction(std::move(newBranch));
          }
          newBlocks.push_back(std::move(bp));
          const std::vector<ir::Operand> label_in_operands;
          std::unique_ptr<ir::Instruction> newLabel(new ir::Instruction(
              SpvOpLabel, 0, returnLabelId, label_in_operands));
          bp.reset(new ir::BasicBlock(std::move(newLabel)));
        }
        // load return value into result id of call, if it exists
        if (returnVarId != 0) {
          auto resId = call_ii->result_id();
          assert(resId != 0);
          std::vector<ir::Operand> load_in_operands;
          load_in_operands.push_back(
              ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                          std::initializer_list<uint32_t>{returnVarId}));
          std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(
              SpvOpLoad, calleeTypeId, resId, load_in_operands));
          bp->AddInstruction(std::move(newLoad));
        }
        // copy remaining instructions from caller block
        auto cii = call_ii;
        cii++;
        for (; cii != call_bi->end(); cii++) {
          std::unique_ptr<ir::Instruction> spv_inst(new ir::Instruction(*cii));
          bp->AddInstruction(std::move(spv_inst));
        }
        // finalize
        newBlocks.push_back(std::move(bp));
      } break;
      default: {
        // copy callee instruction and remap all input Ids
        std::unique_ptr<ir::Instruction> spv_inst(new ir::Instruction(*cpi));
        spv_inst->ForEachInId([&callee2caller, &cpi, this](uint32_t* iid) {
          auto s = callee2caller.find(*iid);
          if (s != callee2caller.end()) {
            *iid = s->second;
          } else if (cpi->IsControlFlow()) {
            ir::Instruction* inst =
                def_use_mgr_->id_to_defs().find(*iid)->second;
            if (inst->opcode() == SpvOpLabel) {
              // forward label reference. allocate a new label id, map it, use
              // it and check for it at each label.
              auto nid = this->getNextId();
              callee2caller[*iid] = nid;
              *iid = nid;
            }
          }
        });
        // map and reset result id
        auto rid = spv_inst->result_id();
        if (rid != 0) {
          auto nid = this->getNextId();
          callee2caller[rid] = nid;
          spv_inst->SetResultId(nid);
        }
        bp->AddInstruction(std::move(spv_inst));
      } break;
    }
  });
}

bool InlinePass::Inline(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (ii->opcode() == SpvOp::SpvOpFunctionCall) {
        // Inline call
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(newBlocks, newVars, ii, bi);
        // update block map given replacement blocks
        for (auto& blk : newBlocks) {
          id2block_[blk->GetLabelId()] = &*blk;
        }
        // update phi functions in succesor blocks if call block
        // is replaced with more than one block
        if (newBlocks.size() > 1) {
          auto firstBlk = newBlocks.begin();
          auto lastBlk = newBlocks.end() - 1;
          uint32_t firstId = (*firstBlk)->GetLabelId();
          uint32_t lastId = (*lastBlk)->GetLabelId();
          (*lastBlk)->ForEachSucc([&firstId, &lastId, this](uint32_t succ) {
            ir::BasicBlock* sbp = this->id2block_[succ];
            sbp->ForEachPhiInst([&firstId, &lastId](ir::Instruction* phi) {
              phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
                if (*id == firstId) *id = lastId;
              });
            });
          });
        }
        // replace old calling block with new block(s)
        bi = bi.Erase();
        bi = bi.MoveBefore(newBlocks);
        // insert new function variables
        if (newVars.size() > 0) {
          auto vbi = func->begin();
          auto vii = vbi->begin();
          vii.MoveBefore(newVars);
        }
        // restart inlining at beginning of calling block
        ii = bi->begin();
        modified = true;
      } else {
        ii++;
      }
    }
  }
  return modified;
}

void InlinePass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id
  nextId_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    nextId_ = std::max(nextId_, id_def.first);
  }
  nextId_++;

  module_ = module;

  // initialize function and block maps
  id2function_.clear();
  id2block_.clear();
  for (auto& fn : *module_) {
    id2function_[fn.GetResultId()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.GetLabelId()] = &blk;
    }
  }
};

Pass::Status InlinePass::ProcessImpl() {
  // do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetOperand(SPV_ENTRY_POINT_FUNCTION_ID).words[0]];
    modified = modified || Inline(fn);
  }

  finalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlinePass::InlinePass()
    : nextId_(0), module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status InlinePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
