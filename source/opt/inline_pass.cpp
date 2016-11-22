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

namespace spvtools {
namespace opt {

// Generate callee code into newBlocks to be inlined for the function call at ii.
// Also add new function variables into caller func

void InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<ir::BasicBlock>>& newBlocks,
    ir::UptrVectorIterator<ir::Instruction> ii,
    ir::UptrVectorIterator<ir::BasicBlock> bi,
    ir::Function* func) {

  // Map from callee id to caller id
  std::unordered_map<uint32_t, uint32_t> inline2func;

  uint32_t calleeId = ii->GetOperand(SPV_FUNCTION_CALL_FUNCTION_ID).words[0];
  ir::Function* calleeFn = id2function[calleeId];

  // Map parameters to actual arguments
  int i = 0;
  calleeFn->ForEachParam([&ii,&i,&inline2func](const ir::Instruction* cpi) {
    auto pid = cpi->GetOperand(SPV_FUNCTION_PARAMETER_RESULT_ID).words[0];
    inline2func[pid] = ii->GetOperand(SPV_FUNCTION_CALL_ARGUMENT_ID + i).words[0];
    i++;
  });

  // Define caller local variables for all callee variables and create map to them
  auto cbi = calleeFn->begin();
  auto cii = cbi->begin();
  std::vector<std::unique_ptr<ir::Instruction>> newVars;
  while (cii->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(*cii));
    uint32_t newId = getNextId();
    var_inst->SetResultId(newId);
    inline2func[cii->result_id()] = newId;
    newVars.push_back(std::move(var_inst));
    cii++;
  }

  // Create return label id
  uint32_t returnLabelId = 0;

  // Create return var if needed
  uint32_t funcTypeId = calleeFn->GetTypeId();
  ir::Instruction *type_inst = def_use_mgr_->id_to_defs().find(funcTypeId)->second;
  uint32_t returnVarId = 0;
  if ( type_inst->opcode() != SpvOpTypeVoid) {
    returnVarId = getNextId();
    std::vector<ir::Operand> in_operands;
    in_operands.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
      std::initializer_list<uint32_t>{uint32_t(SpvStorageClassFunction)});
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(SpvOpVariable,
      funcTypeId, returnVarId, in_operands));
    newVars.push_back(std::move(var_inst));
  }

  // Insert new caller vars into caller function
  if (newVars.size() > 0) {
    auto vbi = func->begin();
    auto vii = vbi->begin();
    vii.MoveBefore(newVars);
  }

  // Clone and map callee code
  bool returned = false;
  std::unique_ptr<ir::BasicBlock> bp;
  calleeFn->ForEachInst([&newBlocks,&inline2func,&bi,&ii,&bp,&returned,&returnLabelId,&returnVarId,this](const ir::Instruction* cpi) {
    switch (cpi->opcode()) {
    case SpvOpFunction:
    case SpvOpFunctionParameter:
    case SpvOpVariable:
      // already processed
      break;
    case SpvOpLabel: {
        // if previous instruction was early return, insert branch instruction
        if (returned) {
          if (returnLabelId == 0)
            returnLabelId = this->getNextId();
          std::vector<ir::Operand> branch_in_operands;
          branch_in_operands.push_back(ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
              std::initializer_list<uint32_t>{returnLabelId}));
          std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
              SpvOpBranch, 0, 0, branch_in_operands));
          bp->AddInstruction(std::move(newBranch));
          returned = false;
        }
        // finish current block (if it exists) and create next block
        uint32_t resId;
        bool firstBlock = false;
        if (bp != nullptr) {
          newBlocks.push_back(std::move(bp));
          resId = this->getNextId();
        }
        else {
          // first block needs to use label of original block
          resId = bi->GetLabelId();
          firstBlock = true;
        }
        const std::vector<ir::Operand> label_in_operands;
        std::unique_ptr<ir::Instruction> newLabel(new ir::Instruction(
            SpvOpLabel, resId, 0, label_in_operands));
        bp.reset(new ir::BasicBlock(std::move(newLabel)));
        if (firstBlock) {
          // Copy contents of original caller block up to call instruction
          for (auto rii = bi->begin(); rii != ii; rii++) {
            std::unique_ptr<ir::Instruction> spv_inst(
                new ir::Instruction(*rii));
            bp->AddInstruction(std::move(spv_inst));
          }
        }
      }
      break;
    case SpvOpReturn: {
        // if return variable created, store return value to it
        if (returnVarId != 0) {
          auto oid = cpi->GetInOperand(SPV_STORE_OBJECT_ID).words[0];
          assert(oid != 0);
          std::vector<ir::Operand> store_in_operands;
          store_in_operands.push_back(ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
              std::initializer_list<uint32_t>{returnVarId}));
          store_in_operands.push_back(ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
              std::initializer_list<uint32_t>{oid}));
          std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
              SpvOpStore, 0, 0, store_in_operands));
          bp->AddInstruction(std::move(newBranch));
        }
        returned = true;
      }
      break;
    case SpvOpFunctionEnd: {
      }
      break;
    default: {
        // copy callee instruction and remap all input Ids
        std::unique_ptr<ir::Instruction> spv_inst(
            new ir::Instruction(*cpi));
        spv_inst->ForEachInId([&inline2func](uint32_t* iid) {
          auto s = inline2func.find(*iid);
          if (s != inline2func.end()) {
            *iid = s->second;
          }
        });
        // map and reset result id
        auto rid = spv_inst->result_id();
        if (rid != 0) {
          auto nid = this->getNextId();
          inline2func[rid] = nid;
          spv_inst->SetResultId(nid);
        }
        bp->AddInstruction(std::move(spv_inst));
      }
      break;
    }
  });

  // clone called function, remapping ids and params, converting returns to store/branches
  // insert clone code before call
  // turn call into load of return var
}

bool InlinePass::Inline(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (ii->opcode() == SpvOp::SpvOpFunctionCall) {
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        GenInlineCode(newBlocks, ii, bi, func);
        bi = bi.Erase();
        bi = bi.MoveBefore(newBlocks);
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
    for (const auto& id_def : def_use_mgr_->id_to_defs()) {
        nextId_ = std::max(nextId_, id_def.first);
    }
    module_ = module;
};

Pass::Status InlinePass::ProcessImpl() {

  // initialize function map
  id2function.clear();
  for (auto& fn : *module_) {
    id2function[fn.GetResultId()] = &fn;
  }

  // do inlining on each entry point function
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn = id2function[e.GetOperand(SPV_ENTRY_POINT_FUNCTION_ID).words[0]];
    modified = modified || Inline(fn);
  }

  finalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlinePass::InlinePass()
    : nextId_(0),
    module_(nullptr),
    def_use_mgr_(nullptr) {}

Pass::Status InlinePass::Process(ir::Module* module) {
    Initialize(module);
    return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
