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

#include "mem_pass.h"

#include "iterator.h"

namespace spvtools {
namespace opt {

bool MemPass::IsMathType(
    const ir::Instruction* typeInst) const {
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

bool MemPass::IsTargetType(
    const ir::Instruction* typeInst) const {
  if (IsMathType(typeInst))
    return true;
  if (typeInst->opcode() == SpvOpTypeArray)
    return IsMathType(def_use_mgr_->GetDef(typeInst->GetSingleWordOperand(1)));
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  // All struct members must be math type
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](const uint32_t* tid) {
    ir::Instruction* compTypeInst = def_use_mgr_->GetDef(*tid);
    if (!IsMathType(compTypeInst)) ++nonMathComp;
  });
  return nonMathComp == 0;
}

MemPass::MemPass() : def_use_mgr_(nullptr) {}

}  // namespace opt
}  // namespace spvtools

