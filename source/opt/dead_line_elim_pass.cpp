// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#include "source/opt/dead_line_elim_pass.h"

#include <set>
#include <unordered_set>
#include <vector>

namespace {

  // Input Operand Indices
  static const int kSpvLineFileInIdx = 0;
  static const int kSpvLineLineInIdx = 1;
  static const int kSpvLineColInIdx = 2;

}  // anonymous namespace

namespace spvtools {
namespace opt {

Pass::Status DeadLineElimPass::Process() {
  bool modified = false;
  uint32_t file_id = 0;
  uint32_t line = 0;
  uint32_t col = 0;
  // Process types, globals, constants
  for (Instruction& inst : get_module()->types_values())
    modified |= EliminateDeadLines(&inst, &file_id, &line, &col);
  // Process functions
  for (Function& function : *get_module()) {
    modified |= EliminateDeadLines(&function.DefInst(), &file_id, &line, &col);
    function.ForEachParam(
        [this, &modified, &file_id, &line, &col](Instruction* param) {
      modified |= EliminateDeadLines(param, &file_id, &line, &col);
    });
    for (BasicBlock& block : function) {
      modified |= EliminateDeadLines(block.GetLabelInst(), &file_id, &line,
                                     &col);
      for (Instruction& inst : block) {
        modified |= EliminateDeadLines(&inst, &file_id, &line, &col);
        // Don't process terminal instruction if preceeded by merge
        if (inst.opcode() == SpvOpSelectionMerge || 
            inst.opcode() == SpvOpLoopMerge)
          break;
      }
      // Nullify line info after each block.
      file_id = 0;
    }
    modified |= EliminateDeadLines(function.EndInst(), &file_id, &line, &col);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool DeadLineElimPass::EliminateDeadLines(Instruction* inst, uint32_t *file_id,
                                          uint32_t *line, uint32_t *col) {
  // If no debug line instructions, return without modifying lines
  if (inst->dbg_line_insts().empty()) return false;
  // Only the last debug instruction needs to be considered; delete all others
  bool modified = inst->dbg_line_insts().size() > 1;
  Instruction last_inst = inst->dbg_line_insts().back();
  inst->dbg_line_insts().clear();
  // If last line is OpNoLine
  if (last_inst.opcode() == SpvOpNoLine) {
    // If no propagated line info, throw away redundant OpNoLine
    if (*file_id == 0) {
      modified = true;
    // Else replace OpNoLine and propagate no line info
    } else {
      inst->dbg_line_insts().push_back(last_inst);
      *file_id = 0;
    }
  // Else last line is OpLine
  } else {
    assert(last_inst.opcode() == SpvOpLine && "unexpected debug inst");
    // If propagated info matches last line, throw away last line
    if (*file_id == last_inst.GetSingleWordInOperand(kSpvLineFileInIdx) &&
        *line == last_inst.GetSingleWordInOperand(kSpvLineLineInIdx) &&
        *col == last_inst.GetSingleWordInOperand(kSpvLineColInIdx)) {
      modified = true;
    // Else replace last line and propagate line info
    } else {
      *file_id = last_inst.GetSingleWordInOperand(kSpvLineFileInIdx);
      *line = last_inst.GetSingleWordInOperand(kSpvLineLineInIdx);
      *col = last_inst.GetSingleWordInOperand(kSpvLineColInIdx);
      inst->dbg_line_insts().push_back(last_inst);
    }
  }
  return modified;
}

}  // namespace opt
}  // namespace spvtools
