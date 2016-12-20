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

#ifndef LIBSPIRV_OPT_SSAMEM_PASS_H_
#define LIBSPIRV_OPT_SSAMEM_PASS_H_


#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <utility>

#include "def_use_manager.h"
#include "module.h"
#include "basic_block.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SSAMemPass : public Pass {
 public:
  SSAMemPass();
  const char* name() const override { return "sroa"; }
  Status Process(ir::Module*) override;

 private:
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;

  // Map from SSA Variable to its single store
  std::unordered_map<uint32_t, ir::Instruction*> ssaVars;

  // Hash for CompKey
  struct pairhash {
    public:
      template <typename T, typename U>
      std::size_t operator()(const std::pair<T, U> &x) const
      {
          return (std::hash<T>()(x.first) << 8) ^ std::hash<U>()(x.second);
      }
  };

  // Map from SSA Component (Var, Index pair) to its single store
  typedef std::pair<uint32_t, uint32_t> CompKey;
  std::unordered_map<CompKey, ir::Instruction*, pairhash> ssaComps;

  // Set of SSA Component Variables
  std::unordered_set<uint32_t> ssaCompVars;

  // Set of non-SSA Variables
  std::unordered_set<uint32_t> nonSsaVars;

  // Set of verified target types
  std::unordered_set<uint32_t> seenTargetTypes;

  // Set of verified target types
  std::unordered_set<uint32_t> seenNonTargetTypes;

  // Returns true if type is a scalar type
  // or a vector or matrix
  bool isMathType(ir::Instruction* typeInst);

  // Returns true if type is a scalar, vector, matrix
  // or struct of only those types
  bool isTargetType(ir::Instruction* typeInst);

  // Next unused ID
  uint32_t nextId_;

  inline void finalizeNextId(ir::Module* module) {
    module->SetIdBound(nextId_);
  }
  inline uint32_t getNextId() { return nextId_++; }

  // Find all function scope variables that are stored to only once
  // and create two maps: one for full variable stores and one for
  // component stores. They will map variable (and component index)
  // to the store value Id. Also cache all variables that
  // are not SSA. Only analyze variables of scalar, vector, 
  // matrix types and struct types containing only these types.
  void SSAMemAnalyze(ir::Function* func);

  // For each load of SSA variable, replace all uses of the load
  // with the value stored, if possible. Assumes that SSAMemAnalyze
  // has just been run for func. Return true if the any
  // instructions are modified.
  bool SSAMemProcess(ir::Function* func);

  // For each load of SSA variable, replace all uses of the load
  // with the value stored, if possible. Return true if the any
  // instructions are modified. 
  bool SSAMem(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  ir::Module* module_;
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSAMEM_PASS_H_

