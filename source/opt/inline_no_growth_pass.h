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

#ifndef LIBSPIRV_OPT_INLINE_NO_GROWTH_PASS_H_
#define LIBSPIRV_OPT_INLINE_NO_GROWTH_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <vector>
#include <unordered_map>

#include "def_use_manager.h"
#include "module.h"
#include "inline_pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlineNoGrowthPass : public InlinePass {

 public:
  InlineNoGrowthPass();
  Status Process(ir::Module*) override;

  const char* name() const override { return "inline-no-growth"; }

 private:
  // Map from function id to inlined size, defined as count of instructions
  // that will likely not be eliminated by inlining or memory optimizations.
  std::unordered_map<uint32_t, uint32_t> funcId2inlinedSize_;

  // Map from function id to call size, defined as the number of stores to
  // parameters plus the size of the call instruction.
  std::unordered_map<uint32_t, uint32_t> funcId2callSize_;

  // Compute funcId2callSize_ for all functions in module_.
  void ComputeCallSize();

  // Compute funcId2inlinedSize_ for all functions in module_. Depends on
  // ComputeCallSize().
  void ComputeInlinedSize();

  // Return true if inlining function call |callInst| will not cause
  // code size to grow.
  bool IsNoGrowthCall(const ir::Instruction* callInst);

  // Inline all function calls in |func| that have opaque params or return
  // type. Inline similarly all code that is inlined into func. Return true
  // if func is modified.
  bool InlineNoGrowth(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INLINE_NO_GROWTH_PASS_H_
