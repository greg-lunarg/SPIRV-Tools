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

#ifndef LIBSPIRV_OPT_RELAX_FLOAT_OPS_PASS_H_
#define LIBSPIRV_OPT_RELAX_FLOAT_OPS_PASS_H_

#include "source/opt/ir_builder.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class RelaxFloatOpsPass : public Pass {
 public:
  RelaxFloatOpsPass() : Pass() {}

  ~RelaxFloatOpsPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "convert-to-half-pass"; }

 private:

   // Return true if |inst| can have the RelaxedPrecision decoration applied
   // to it.
   bool is_relaxable(Instruction* inst);

   // Return base type of |ty_id| type
   Instruction* get_base_type(uint32_t ty_id);

   // Return true if |inst| returns scalar, vector or matrix type with base
   // float and width 32
   bool is_float32(Instruction* inst);

   // Return true if |r_id| is decorated with RelaxedPrecision
   bool is_relaxed(uint32_t r_id);

   // If |inst| is an instruction of float32-based type and is not decorated 
   // RelaxedPrecision, add it's id to ids_to_relax_.
   void ProcessInst(Instruction* inst);

  // Call ProcessInst on every instruction in |func|.
  void ProcessFunction(Function* func);

  // Process all functions in the call tree of the function ids in |roots|.
  void ProcessCallTreeFromRoots(std::queue<uint32_t>* roots);

  Pass::Status ProcessImpl();

  // Initialize state for converting to half
  void Initialize();

  // Map from function id to function pointer.
  std::unordered_map<uint32_t, Function*> id2function_;

  // Set of float result core operations to be processed
  std::unordered_set<uint32_t> target_ops_core_f_rslt;

  // Set of float operand core operations to be processed
  std::unordered_set<uint32_t> target_ops_core_f_opnd;

  // Set of 450 extension operations to be processed
  std::unordered_set<uint32_t> target_ops_450_;

  // Set of sample operations
  std::unordered_set<uint32_t> sample_ops_;

  // GLSL 540 extension id
  uint32_t glsl450_ext_id_;

  // Set of ids to be decorated RelaxedPrecision
  std::unordered_set<uint32_t> ids_to_relax_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_RELAX_FLOAT_OPS_PASS_H_
