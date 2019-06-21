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

#ifndef LIBSPIRV_OPT_CONVERT_TO_HALF_PASS_H_
#define LIBSPIRV_OPT_CONVERT_TO_HALF_PASS_H_

#include "source/opt/ir_builder.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class ConvertToHalfPass : public Pass {
 public:
  ConvertToHalfPass() : Pass() {}

  ~ConvertToHalfPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "convert-to-half-pass"; }

 private:
   // Return true if |inst| is an arithmetic op that can be of type float16
   bool is_arithmetic(Instruction* inst);

   // Return base type of |ty_id| type
   Instruction* get_base_type(uint32_t ty_id);

   // Return true if |inst| returns scalar, vector or matrix type with base
   // float and |width|
   bool is_float(Instruction* inst, uint32_t width);

   // Return true if |inst| is decorated with RelaxedPrecision
   bool is_relaxed(Instruction* inst);

   // Return equivalent to float type |ty_id| with |width|
   uint32_t get_equiv_float_ty_id(uint32_t ty_id, uint32_t width);

   void GenConvert(uint32_t ty_id, uint32_t width, uint32_t* val_idp, InstructionBuilder* builder);

   // If |inst| is a gpu instruction of float type, append to |new_insts|
   // the result of relaxing its precision to half. Specifically, generate code
   // to convert any operands to half, execute the instruction with type half,
   // and convert the result back to float.
   bool GenHalfCode(Instruction* inst);

   // If |inst| is an FConvert of a matrix type, decompose it to a series
   // of vector extracts, converts and inserts into an Undef. These are generated
   // by GenHalfCode because they are easier to optimize, but we need to clean
   // them up before leaving.
   bool MatConvertCleanup(Instruction* inst);

  // Call GenHalfCode on every instruction in |func|.
  // If code is generated for an instruction, replace the instruction
  // with the new instructions that are generated.
  bool ProcessFunction(Function* func);

  // Process all functions in the call tree of the function ids in |roots|.
  bool ProcessCallTreeFromRoots(
    std::queue<uint32_t>* roots);

  Pass::Status ProcessImpl();

  // Initialize state for converting to half
  void Initialize();

  // Map from function id to function pointer.
  std::unordered_map<uint32_t, Function*> id2function_;

  // Set of core operations to be processed
  std::unordered_set<uint32_t> target_ops_core_;

  // Set of 450 extension operations to be processed
  std::unordered_set<uint32_t> target_ops_450_;

  // Set of sample operations
  std::unordered_set<uint32_t> sample_ops_;

  // Set of dref sample operations
  std::unordered_set<uint32_t> dref_sample_ops_;

  // GLSL 540 extension id
  uint32_t glsl450_ext_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_CONVERT_TO_HALF_PASS_H_
