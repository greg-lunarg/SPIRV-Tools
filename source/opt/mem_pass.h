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

#ifndef LIBSPIRV_OPT_OPT_PASS_H_
#define LIBSPIRV_OPT_OPT_PASS_H_


#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class MemPass : public Pass {
 public:
  MemPass();
  virtual ~MemPass() = default;

 protected:
  // Returns true if |typeInst| is a scalar type
  // or a vector or matrix
  bool IsMathType(const ir::Instruction* typeInst) const;

  // Returns true if |typeInst| is a math type or a struct or array
  // of a math type.
  // TODO(): Add more complex types to convert
  bool IsTargetType(const ir::Instruction* typeInst) const;

  // Returns true if |opcode| is a non-ptr access chain op
  bool IsNonPtrAccessChain(const SpvOp opcode) const;

  // Given a load or store |ip|, return the pointer instruction.
  // Also return the base variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if |varId| is a previously identified target variable.
  // Return false if |varId| is a previously identified non-target variable.
  // See FindTargetVars() for definition of target variable. If variable is
  // not cached, return true if variable is a function scope variable of
  // target type, false otherwise. Updates caches of target and non-target
  // variables.
  bool IsTargetVar(uint32_t varId);

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Cache of verified target vars
  std::unordered_set<uint32_t> seen_target_vars_;

  // Cache of verified non-target vars
  std::unordered_set<uint32_t> seen_non_target_vars_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_OPT_PASS_H_

