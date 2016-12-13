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

#ifndef LIBSPIRV_OPT_SROA_PASS_H_
#define LIBSPIRV_OPT_SROA_PASS_H_

#include <algorithm>

#include "def_use_manager.h"
#include "module.h"
#include "basic_block.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SRoAPass : public Pass {
 public:
  SRoAPass();
  const char* name() const override { return "sroa"; }
  Status Process(ir::Module*) override;

 private:
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;

  // Map from block's label id to block
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block;

  // Next unused ID
  uint32_t nextId_;

  inline void finalizeNextId(ir::Module* module) {
    module->SetIdBound(nextId_);
  }
  inline uint32_t getNextId() { return nextId_++; }

  // Perform Scalar Replacement of Aggregates on func.
  //
  // Specifically, transform all non-indexed load/store pairs of
  // targeted structs to the equivalent loads and stores of
  // their components. Target structs of function scope whose
  // components are only scalars, vectors and matrices.
  //
  // Also transform all indexed loads and stores into vector and
  // matrix types into the equivalent load or store of the vector
  // or matrix with an extract or insert.
  //
  // This transform is generally intended to remove all aliasing
  // between loads and stores for the targeted structs and ease
  // store/load elimination in succeeding passes.
  bool SRoA(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Return true if typeId is struct of all scalar, vector or
  // matrix components
  bool IsStructOfScalar(ir::Instruction* typeInst);

  ir::Module* module_;
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from structure load id to first component load id 
  std::unordered_map<uint32_t, uint32_t> struct2comp_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SROA_PASS_H_
