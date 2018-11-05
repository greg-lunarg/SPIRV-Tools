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

#include "source/opt/prop_lines_pass.h"

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

Pass::Status PropagateLinesPass::Process() {
  bool modified = false;
  uint32_t file_id = 0;
  uint32_t line = 0;
  uint32_t col = 0;
  // Process types, globals, constants
  for (Instruction& inst : get_module()->types_values())
    modified |= PropagateLine(&inst, &file_id, &line, &col);
  // Process functions
  for (Function& function : *get_module()) {
    modified |= PropagateLine(&function.DefInst(), &file_id, &line, &col);
    function.ForEachParam(
        [this, &modified, &file_id, &line, &col](Instruction* param) {
      modified |= PropagateLine(param, &file_id, &line, &col);
    });
    for (BasicBlock& block : function) {
      modified |= PropagateLine(block.GetLabelInst(), &file_id, &line, &col);
      for (Instruction& inst : block) {
        modified |= PropagateLine(&inst, &file_id, &line, &col);
        // Don't process terminal instruction if preceeded by merge
        if (inst.opcode() == SpvOpSelectionMerge || 
            inst.opcode() == SpvOpLoopMerge)
          break;
      }
      // Nullify line info after each block.
      file_id = 0;
    }
    modified |= PropagateLine(function.EndInst(), &file_id, &line, &col);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool PropagateLinesPass::PropagateLine(Instruction* inst, uint32_t *file_id,
                                       uint32_t *line, uint32_t *col) {
  bool modified = false;
  // only the last debug instruction needs to be considered
  auto line_itr = inst->dbg_line_insts().rbegin();
  // if no line instructions, propagate previous info
  if (line_itr == inst->dbg_line_insts().rend()) {
    // if no current line info, add OpNoLine, else OpLine
    if (*file_id == 0)
      inst->dbg_line_insts().push_back(Instruction(context(), SpvOpNoLine));
    else
      inst->dbg_line_insts().push_back(Instruction(
        context(), SpvOpLine, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {*file_id} },
         {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {*line} },
         {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {*col}}}));
    modified = true;
  // else pre-existing line instruction, so update source line info
  } else {
    if (line_itr->opcode() == SpvOpNoLine) {
      *file_id = 0;
    } else {
      assert(line_itr->opcode() == SpvOpLine && "unexpected debug inst");
      *file_id = line_itr->GetSingleWordInOperand(kSpvLineFileInIdx);
      *line = line_itr->GetSingleWordInOperand(kSpvLineLineInIdx);
      *col = line_itr->GetSingleWordInOperand(kSpvLineColInIdx);
    }
  }
  return modified;
}

}  // namespace opt
}  // namespace spvtools
