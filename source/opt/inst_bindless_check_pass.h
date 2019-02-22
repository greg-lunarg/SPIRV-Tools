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

#ifndef LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_
#define LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_

#include "instrument_pass.h"

namespace spvtools {
namespace opt {

// This class/pass is designed to support the bindless (descriptor indexing)
// GPU-assisted validation layer of
// https://github.com/KhronosGroup/Vulkan-ValidationLayers. Its internal and
// external design may change as the layer evolves.
class InstBindlessCheckPass : public InstrumentPass {
 public:
  // For test harness only
  InstBindlessCheckPass()
      : InstrumentPass(7, 23, kInstValidationIdBindless),
        input_length_enabled_(true),
        input_init_enabled_(true) {}
  // For all other interfaces
  InstBindlessCheckPass(uint32_t desc_set, uint32_t shader_id,
                        bool input_length_enable, bool input_init_enable)
      : InstrumentPass(desc_set, shader_id, kInstValidationIdBindless),
        input_length_enabled_(input_length_enable),
        input_init_enabled_(input_init_enable) {}

  ~InstBindlessCheckPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "inst-bindless-check-pass"; }

 private:
  // Generate instructions into |builder| to read length of runtime descriptor
  // array |var_id| from debug input buffer and return id of value.
  uint32_t GenDebugReadLength(uint32_t var_id, InstructionBuilder* builder);

  // Generate instructions into |builder| to read initialization status of descriptor
  // array |image_id| at |index_id| from debug input buffer and return id of value.
  uint32_t GenDebugReadInit(uint32_t image_id, uint32_t index_id,
      InstructionBuilder* builder);
  
  // Clone original reference
  uint32_t CloneOriginalReference(
      BasicBlock::iterator ref_inst_itr,
      Instruction* desc_load_inst,
      Instruction* image_inst,
      uint32_t image_id, InstructionBuilder* builder);

  // Initialize state for instrumenting bindless checking
  void InitializeInstBindlessCheck();

  // These functions do bindless checking instrumentation on a single
  // instruction which references through a descriptor. GenBoundsCheckCode
  // checks that a index into a descriptor array is in-bounds. GenInitCheckCode
  // checks that the descriptor has been initialized if the
  // SPV_EXT_descriptor_indexing extension is enabled.
  //
  // The functions are designed to be passed to
  // InstrumentPass::InstProcessEntryPointCallTree(), which applies the
  // function to each instruction in a module and replaces the instruction
  // if warranted.
  //
  // If |ref_inst_itr| is a bindless reference, return in |new_blocks| the
  // result of instrumenting it with validation code within its block at
  // |ref_block_itr|.  The validation code first executes a check for the
  // specific condition called for. If the check passes, it executes
  // the remainder of the reference, otherwise writes a record to the debug
  // output buffer stream including |function_idx, instruction_idx, stage_idx|
  // and replaces the reference with the null value of the original type. The
  // block at |ref_block_itr| can just be replaced with the blocks in
  // |new_blocks|, which will contain at least two blocks. The last block will
  // comprise all instructions following |ref_inst_itr|,
  // preceded by a phi instruction.
  //
  // These instrumentation functions utilize GenDebugDirectRead() to read data
  // from the debug input buffer, specifically the lengths of variable length
  // descriptor arrays, and the initialization status of each descriptor.
  // The format of the debug input buffer is documented in instrument.hpp.
  //
  // These instrumentation functions utilize GenDebugStreamWrite() to write its
  // error records. The validation-specific part of the error record will
  // have the format:
  //
  //    Validation Error Code (=kInstErrorBindlessBounds)
  //    Descriptor Index
  //    Descriptor Array Size
  //
  // The Descriptor Index is the index which has been determined to be
  // out-of-bounds.
  //
  // The Descriptor Array Size is the size of the descriptor array which was
  // indexed.
  void GenBoundsCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr,
      uint32_t stage_idx, std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  void GenInitCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr,
      uint32_t stage_idx, std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  Pass::Status ProcessImpl();

  // True if VK_EXT_descriptor_indexing is defined
  bool ext_descriptor_indexing_defined_;

  // Enable instrumentation of runtime array length checking
  bool input_length_enabled_;

  // Enable instrumentation of descriptor initialization checking
  bool input_init_enabled_;

  // Mapping from variable to descriptor set
  std::unordered_map<uint32_t, uint32_t> var2desc_set_;

  // Mapping from variable to binding
  std::unordered_map<uint32_t, uint32_t> var2binding_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_
