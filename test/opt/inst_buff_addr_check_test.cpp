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

// Bindless Check Instrumentation Tests.
// Tests ending with V2 use version 2 record format.

#include <string>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InstBuffAddrTest = PassTest<::testing::Test>;

TEST_F(InstBuffAddrTest, InstPhysicalStorageBufferStore) {
  // #version 450
  // #extension GL_EXT_buffer_reference : enable
  // 
  // layout(buffer_reference, buffer_reference_align = 16) buffer bufStruct;
  // 
  // layout(set = 0, binding = 0) uniform ufoo {
  //     bufStruct data;
  //     uint offset;
  // } u_info;
  // 
  // layout(buffer_reference, std140) buffer bufStruct {
  //     layout(offset = 0) int a[2];
  //     layout(offset = 32) int b;
  // };
  // 
  // void main() {
  //     u_info.data.b = 0xca7;
  // }

  const std::string defs_before =
      R"(OpCapability Shader
OpCapability PhysicalStorageBufferAddressesEXT
OpExtension "SPV_EXT_physical_storage_buffer"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel PhysicalStorageBuffer64EXT GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 450
OpSourceExtension "GL_EXT_buffer_reference"
OpName %main "main"
OpName %ufoo "ufoo"
OpMemberName %ufoo 0 "data"
OpMemberName %ufoo 1 "offset"
OpName %bufStruct "bufStruct"
OpMemberName %bufStruct 0 "a"
OpMemberName %bufStruct 1 "b"
OpName %u_info "u_info"
OpMemberDecorate %ufoo 0 Offset 0
OpMemberDecorate %ufoo 1 Offset 8
OpDecorate %ufoo Block
OpDecorate %_arr_int_uint_2 ArrayStride 16
OpMemberDecorate %bufStruct 0 Offset 0
OpMemberDecorate %bufStruct 1 Offset 32
OpDecorate %bufStruct Block
OpDecorate %u_info DescriptorSet 0
OpDecorate %u_info Binding 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
OpTypeForwardPointer %_ptr_PhysicalStorageBufferEXT_bufStruct PhysicalStorageBufferEXT
%uint = OpTypeInt 32 0
%ufoo = OpTypeStruct %_ptr_PhysicalStorageBufferEXT_bufStruct %uint
%int = OpTypeInt 32 1
%uint_2 = OpConstant %uint 2
%_arr_int_uint_2 = OpTypeArray %int %uint_2
%bufStruct = OpTypeStruct %_arr_int_uint_2 %int
%_ptr_PhysicalStorageBufferEXT_bufStruct = OpTypePointer PhysicalStorageBufferEXT %bufStruct
%_ptr_Uniform_ufoo = OpTypePointer Uniform %ufoo
%u_info = OpVariable %_ptr_Uniform_ufoo Uniform
%int_0 = OpConstant %int 0
%_ptr_Uniform__ptr_PhysicalStorageBufferEXT_bufStruct = OpTypePointer Uniform %_ptr_PhysicalStorageBufferEXT_bufStruct
%int_1 = OpConstant %int 1
%int_3239 = OpConstant %int 3239
%_ptr_PhysicalStorageBufferEXT_int = OpTypePointer PhysicalStorageBufferEXT %int
)";

  const std::string defs_after =
      R"(OpCapability Shader
OpCapability PhysicalStorageBufferAddressesEXT
OpCapability Int64
OpExtension "SPV_EXT_physical_storage_buffer"
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel PhysicalStorageBuffer64EXT GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 450
OpSourceExtension "GL_EXT_buffer_reference"
OpName %main "main"
OpName %ufoo "ufoo"
OpMemberName %ufoo 0 "data"
OpMemberName %ufoo 1 "offset"
OpName %bufStruct "bufStruct"
OpMemberName %bufStruct 0 "a"
OpMemberName %bufStruct 1 "b"
OpName %u_info "u_info"
OpMemberDecorate %ufoo 0 Offset 0
OpMemberDecorate %ufoo 1 Offset 8
OpDecorate %ufoo Block
OpDecorate %_arr_int_uint_2 ArrayStride 16
OpMemberDecorate %bufStruct 0 Offset 0
OpMemberDecorate %bufStruct 1 Offset 32
OpDecorate %bufStruct Block
OpDecorate %u_info DescriptorSet 0
OpDecorate %u_info Binding 0
OpDecorate %_runtimearr_ulong ArrayStride 8
OpDecorate %_struct_40 Block
OpMemberDecorate %_struct_40 0 Offset 0
OpDecorate %42 DescriptorSet 7
OpDecorate %42 Binding 2
OpDecorate %_runtimearr_uint ArrayStride 4
OpDecorate %_struct_78 Block
OpMemberDecorate %_struct_78 0 Offset 0
OpMemberDecorate %_struct_78 1 Offset 4
OpDecorate %80 DescriptorSet 7
OpDecorate %80 Binding 0
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
%void = OpTypeVoid
%8 = OpTypeFunction %void
OpTypeForwardPointer %_ptr_PhysicalStorageBufferEXT_bufStruct PhysicalStorageBufferEXT
%uint = OpTypeInt 32 0
%ufoo = OpTypeStruct %_ptr_PhysicalStorageBufferEXT_bufStruct %uint
%int = OpTypeInt 32 1
%uint_2 = OpConstant %uint 2
%_arr_int_uint_2 = OpTypeArray %int %uint_2
%bufStruct = OpTypeStruct %_arr_int_uint_2 %int
%_ptr_PhysicalStorageBufferEXT_bufStruct = OpTypePointer PhysicalStorageBufferEXT %bufStruct
%_ptr_Uniform_ufoo = OpTypePointer Uniform %ufoo
%u_info = OpVariable %_ptr_Uniform_ufoo Uniform
%int_0 = OpConstant %int 0
%_ptr_Uniform__ptr_PhysicalStorageBufferEXT_bufStruct = OpTypePointer Uniform %_ptr_PhysicalStorageBufferEXT_bufStruct
%int_1 = OpConstant %int 1
%int_3239 = OpConstant %int 3239
%_ptr_PhysicalStorageBufferEXT_int = OpTypePointer PhysicalStorageBufferEXT %int
%ulong = OpTypeInt 64 0
%uint_4 = OpConstant %uint 4
%bool = OpTypeBool
%28 = OpTypeFunction %bool %ulong %uint
%uint_1 = OpConstant %uint 1
%_runtimearr_ulong = OpTypeRuntimeArray %ulong
%_struct_40 = OpTypeStruct %_runtimearr_ulong
%_ptr_StorageBuffer__struct_40 = OpTypePointer StorageBuffer %_struct_40
%42 = OpVariable %_ptr_StorageBuffer__struct_40 StorageBuffer
%_ptr_StorageBuffer_ulong = OpTypePointer StorageBuffer %ulong
%uint_0 = OpConstant %uint 0
%uint_32 = OpConstant %uint 32
%71 = OpTypeFunction %void %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_78 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_78 = OpTypePointer StorageBuffer %_struct_78
%80 = OpVariable %_ptr_StorageBuffer__struct_78 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_23 = OpConstant %uint 23
%uint_5 = OpConstant %uint 5
%uint_3 = OpConstant %uint 3
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%uint_9 = OpConstant %uint 9
%uint_48 = OpConstant %uint 48
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %3
%5 = OpLabel
%17 = OpAccessChain %_ptr_Uniform__ptr_PhysicalStorageBufferEXT_bufStruct %u_info %int_0
%18 = OpLoad %_ptr_PhysicalStorageBufferEXT_bufStruct %17
%22 = OpAccessChain %_ptr_PhysicalStorageBufferEXT_int %18 %int_1
OpStore %22 %int_3239 Aligned 16
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %8
%19 = OpLabel
%20 = OpAccessChain %_ptr_Uniform__ptr_PhysicalStorageBufferEXT_bufStruct %u_info %int_0
%21 = OpLoad %_ptr_PhysicalStorageBufferEXT_bufStruct %20
%22 = OpAccessChain %_ptr_PhysicalStorageBufferEXT_int %21 %int_1
%24 = OpConvertPtrToU %ulong %22
%62 = OpFunctionCall %bool %26 %24 %uint_4
OpSelectionMerge %63 None
OpBranchConditional %62 %64 %65
%64 = OpLabel
OpStore %22 %int_3239 Aligned 16
OpBranch %63
%65 = OpLabel
%66 = OpUConvert %uint %24
%68 = OpShiftRightLogical %ulong %24 %uint_32
%69 = OpUConvert %uint %68
%125 = OpFunctionCall %void %70 %uint_48 %uint_2 %66 %69
OpBranch %63
%63 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string new_funcs =
      R"(%26 = OpFunction %bool None %28
%29 = OpFunctionParameter %ulong
%30 = OpFunctionParameter %uint
%31 = OpLabel
OpBranch %32
%32 = OpLabel
%35 = OpPhi %uint %uint_1 %31 %36 %33
OpLoopMerge %38 %33 None
OpBranch %33
%33 = OpLabel
%36 = OpIAdd %uint %35 %uint_1
%45 = OpAccessChain %_ptr_StorageBuffer_ulong %42 %uint_0 %36
%46 = OpLoad %ulong %45
%47 = OpUGreaterThan %bool %46 %29
OpBranchConditional %47 %38 %32
%38 = OpLabel
%48 = OpISub %uint %36 %uint_1
%49 = OpAccessChain %_ptr_StorageBuffer_ulong %42 %uint_0 %48
%50 = OpLoad %ulong %49
%51 = OpISub %ulong %29 %50
%52 = OpUConvert %ulong %30
%53 = OpIAdd %ulong %51 %52
%54 = OpAccessChain %_ptr_StorageBuffer_ulong %42 %uint_0 %uint_0
%55 = OpLoad %ulong %54
%56 = OpUConvert %uint %55
%57 = OpISub %uint %48 %uint_1
%58 = OpIAdd %uint %57 %56
%59 = OpAccessChain %_ptr_StorageBuffer_ulong %42 %uint_0 %58
%60 = OpLoad %ulong %59
%61 = OpULessThanEqual %bool %53 %60
OpReturnValue %61
OpFunctionEnd
%70 = OpFunction %void None %71
%72 = OpFunctionParameter %uint
%73 = OpFunctionParameter %uint
%74 = OpFunctionParameter %uint
%75 = OpFunctionParameter %uint
%76 = OpLabel
%82 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_0
%84 = OpAtomicIAdd %uint %82 %uint_4 %uint_0 %uint_10
%85 = OpIAdd %uint %84 %uint_10
%86 = OpArrayLength %uint %80 1
%87 = OpULessThanEqual %bool %85 %86
OpSelectionMerge %88 None
OpBranchConditional %87 %89 %88
%89 = OpLabel
%90 = OpIAdd %uint %84 %uint_0
%91 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %90
OpStore %91 %uint_10
%93 = OpIAdd %uint %84 %uint_1
%94 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %93
OpStore %94 %uint_23
%95 = OpIAdd %uint %84 %uint_2
%96 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %95
OpStore %96 %72
%99 = OpIAdd %uint %84 %uint_3
%100 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %99
OpStore %100 %uint_5
%104 = OpLoad %v3uint %gl_GlobalInvocationID
%105 = OpCompositeExtract %uint %104 0
%106 = OpCompositeExtract %uint %104 1
%107 = OpCompositeExtract %uint %104 2
%108 = OpIAdd %uint %84 %uint_4
%109 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %108
OpStore %109 %105
%110 = OpIAdd %uint %84 %uint_5
%111 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %110
OpStore %111 %106
%113 = OpIAdd %uint %84 %uint_6
%114 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %113
OpStore %114 %107
%116 = OpIAdd %uint %84 %uint_7
%117 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %116
OpStore %117 %73
%119 = OpIAdd %uint %84 %uint_8
%120 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %119
OpStore %120 %74
%122 = OpIAdd %uint %84 %uint_9
%123 = OpAccessChain %_ptr_StorageBuffer_uint %80 %uint_1 %122
OpStore %123 %75
OpBranch %88
%88 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBuffAddrCheckPass>(
      defs_before + func_before, defs_after + func_after + new_funcs, true,
      true, 7u, 23u, 2u);
}

TEST_F(InstBuffAddrTest, InstPhysicalStorageBufferLoadAndStore) {
  // #version 450
  // #extension GL_EXT_buffer_reference : enable

  // // forward reference
  // layout(buffer_reference) buffer blockType;

  // layout(buffer_reference, std430, buffer_reference_align = 16) buffer blockType {
  //   int x;
  //   blockType next;
  // };

  // layout(std430) buffer rootBlock {
  //   blockType root;
  // } r;

  // void main()
  // {
  //   blockType b = r.root;
  //   b = b.next;
  //   b.x = 531;
  // }

  const std::string defs_before =
    R"(OpCapability Shader
OpCapability PhysicalStorageBufferAddressesEXT
OpExtension "SPV_EXT_physical_storage_buffer"
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel PhysicalStorageBuffer64EXT GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 450
OpSourceExtension "GL_EXT_buffer_reference"
OpName %main "main"
OpName %blockType "blockType"
OpMemberName %blockType 0 "x"
OpMemberName %blockType 1 "next"
OpName %rootBlock "rootBlock"
OpMemberName %rootBlock 0 "root"
OpName %r "r"
OpMemberDecorate %blockType 0 Offset 0
OpMemberDecorate %blockType 1 Offset 8
OpDecorate %blockType Block
OpMemberDecorate %rootBlock 0 Offset 0
OpDecorate %rootBlock Block
OpDecorate %r DescriptorSet 0
OpDecorate %r Binding 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
OpTypeForwardPointer %_ptr_PhysicalStorageBufferEXT_blockType PhysicalStorageBufferEXT
%int = OpTypeInt 32 1
%blockType = OpTypeStruct %int %_ptr_PhysicalStorageBufferEXT_blockType
%_ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer PhysicalStorageBufferEXT %blockType
%rootBlock = OpTypeStruct %_ptr_PhysicalStorageBufferEXT_blockType
%_ptr_StorageBuffer_rootBlock = OpTypePointer StorageBuffer %rootBlock
%r = OpVariable %_ptr_StorageBuffer_rootBlock StorageBuffer
%int_0 = OpConstant %int 0
%_ptr_StorageBuffer__ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer StorageBuffer %_ptr_PhysicalStorageBufferEXT_blockType
%int_1 = OpConstant %int 1
%_ptr_PhysicalStorageBufferEXT__ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer PhysicalStorageBufferEXT %_ptr_PhysicalStorageBufferEXT_blockType
%int_531 = OpConstant %int 531
%_ptr_PhysicalStorageBufferEXT_int = OpTypePointer PhysicalStorageBufferEXT %int
)";

  const std::string defs_after =
    R"(OpCapability Shader
OpCapability PhysicalStorageBufferAddressesEXT
OpCapability Int64
OpCapability Int64
OpExtension "SPV_EXT_physical_storage_buffer"
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel PhysicalStorageBuffer64EXT GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 450
OpSourceExtension "GL_EXT_buffer_reference"
OpName %main "main"
OpName %blockType "blockType"
OpMemberName %blockType 0 "x"
OpMemberName %blockType 1 "next"
OpName %rootBlock "rootBlock"
OpMemberName %rootBlock 0 "root"
OpName %r "r"
OpMemberDecorate %blockType 0 Offset 0
OpMemberDecorate %blockType 1 Offset 8
OpDecorate %blockType Block
OpMemberDecorate %rootBlock 0 Offset 0
OpDecorate %rootBlock Block
OpDecorate %r DescriptorSet 0
OpDecorate %r Binding 0
OpDecorate %_runtimearr_ulong ArrayStride 8
OpDecorate %_struct_46 Block
OpMemberDecorate %_struct_46 0 Offset 0
OpDecorate %48 DescriptorSet 7
OpDecorate %48 Binding 2
OpDecorate %_runtimearr_uint ArrayStride 4
OpDecorate %_struct_85 Block
OpMemberDecorate %_struct_85 0 Offset 0
OpMemberDecorate %_struct_85 1 Offset 4
OpDecorate %87 DescriptorSet 7
OpDecorate %87 Binding 0
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
%void = OpTypeVoid
%3 = OpTypeFunction %void
OpTypeForwardPointer %_ptr_PhysicalStorageBufferEXT_blockType PhysicalStorageBufferEXT
%int = OpTypeInt 32 1
%blockType = OpTypeStruct %int %_ptr_PhysicalStorageBufferEXT_blockType
%_ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer PhysicalStorageBufferEXT %blockType
%rootBlock = OpTypeStruct %_ptr_PhysicalStorageBufferEXT_blockType
%_ptr_StorageBuffer_rootBlock = OpTypePointer StorageBuffer %rootBlock
%r = OpVariable %_ptr_StorageBuffer_rootBlock StorageBuffer
%int_0 = OpConstant %int 0
%_ptr_StorageBuffer__ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer StorageBuffer %_ptr_PhysicalStorageBufferEXT_blockType
%int_1 = OpConstant %int 1
%_ptr_PhysicalStorageBufferEXT__ptr_PhysicalStorageBufferEXT_blockType = OpTypePointer PhysicalStorageBufferEXT %_ptr_PhysicalStorageBufferEXT_blockType
%int_531 = OpConstant %int 531
%_ptr_PhysicalStorageBufferEXT_int = OpTypePointer PhysicalStorageBufferEXT %int
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%ulong = OpTypeInt 64 0
%uint_8 = OpConstant %uint 8
%bool = OpTypeBool
%34 = OpTypeFunction %bool %ulong %uint
%uint_1 = OpConstant %uint 1
%_runtimearr_ulong = OpTypeRuntimeArray %ulong
%_struct_46 = OpTypeStruct %_runtimearr_ulong
%_ptr_StorageBuffer__struct_46 = OpTypePointer StorageBuffer %_struct_46
%48 = OpVariable %_ptr_StorageBuffer__struct_46 StorageBuffer
%_ptr_StorageBuffer_ulong = OpTypePointer StorageBuffer %ulong
%uint_0 = OpConstant %uint 0
%uint_32 = OpConstant %uint 32
%78 = OpTypeFunction %void %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_85 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_85 = OpTypePointer StorageBuffer %_struct_85
%87 = OpVariable %_ptr_StorageBuffer__struct_85 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_4 = OpConstant %uint 4
%uint_23 = OpConstant %uint 23
%uint_5 = OpConstant %uint 5
%uint_3 = OpConstant %uint 3
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_9 = OpConstant %uint 9
%uint_44 = OpConstant %uint 44
%133 = OpConstantNull %_ptr_PhysicalStorageBufferEXT_blockType
%uint_46 = OpConstant %uint 46
)";

  const std::string func_before =
    R"(%main = OpFunction %void None %3
%5 = OpLabel
%16 = OpAccessChain %_ptr_StorageBuffer__ptr_PhysicalStorageBufferEXT_blockType %r %int_0
%17 = OpLoad %_ptr_PhysicalStorageBufferEXT_blockType %16
%21 = OpAccessChain %_ptr_PhysicalStorageBufferEXT__ptr_PhysicalStorageBufferEXT_blockType %17 %int_1
%22 = OpLoad %_ptr_PhysicalStorageBufferEXT_blockType %21 Aligned 8
%26 = OpAccessChain %_ptr_PhysicalStorageBufferEXT_int %22 %int_0
OpStore %26 %int_531 Aligned 16
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
    R"(%main = OpFunction %void None %3
%5 = OpLabel
%16 = OpAccessChain %_ptr_StorageBuffer__ptr_PhysicalStorageBufferEXT_blockType %r %int_0
%17 = OpLoad %_ptr_PhysicalStorageBufferEXT_blockType %16
%21 = OpAccessChain %_ptr_PhysicalStorageBufferEXT__ptr_PhysicalStorageBufferEXT_blockType %17 %int_1
%30 = OpConvertPtrToU %ulong %21
%68 = OpFunctionCall %bool %32 %30 %uint_8
OpSelectionMerge %69 None
OpBranchConditional %68 %70 %71
%70 = OpLabel
%72 = OpLoad %_ptr_PhysicalStorageBufferEXT_blockType %21 Aligned 8
OpBranch %69
%71 = OpLabel
%73 = OpUConvert %uint %30
%75 = OpShiftRightLogical %ulong %30 %uint_32
%76 = OpUConvert %uint %75
%132 = OpFunctionCall %void %77 %uint_44 %uint_2 %73 %76
OpBranch %69
%69 = OpLabel
%134 = OpPhi %_ptr_PhysicalStorageBufferEXT_blockType %72 %70 %133 %71
%26 = OpAccessChain %_ptr_PhysicalStorageBufferEXT_int %134 %int_0
%135 = OpConvertPtrToU %ulong %26
%136 = OpFunctionCall %bool %32 %135 %uint_4
OpSelectionMerge %137 None
OpBranchConditional %136 %138 %139
%138 = OpLabel
OpStore %26 %int_531 Aligned 16
OpBranch %137
%139 = OpLabel
%140 = OpUConvert %uint %135
%141 = OpShiftRightLogical %ulong %135 %uint_32
%142 = OpUConvert %uint %141
%144 = OpFunctionCall %void %77 %uint_46 %uint_2 %140 %142
OpBranch %137
%137 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string new_funcs =
    R"(%32 = OpFunction %bool None %34
%35 = OpFunctionParameter %ulong
%36 = OpFunctionParameter %uint
%37 = OpLabel
OpBranch %38
%38 = OpLabel
%41 = OpPhi %uint %uint_1 %37 %42 %39
OpLoopMerge %44 %39 None
OpBranch %39
%39 = OpLabel
%42 = OpIAdd %uint %41 %uint_1
%51 = OpAccessChain %_ptr_StorageBuffer_ulong %48 %uint_0 %42
%52 = OpLoad %ulong %51
%53 = OpUGreaterThan %bool %52 %35
OpBranchConditional %53 %44 %38
%44 = OpLabel
%54 = OpISub %uint %42 %uint_1
%55 = OpAccessChain %_ptr_StorageBuffer_ulong %48 %uint_0 %54
%56 = OpLoad %ulong %55
%57 = OpISub %ulong %35 %56
%58 = OpUConvert %ulong %36
%59 = OpIAdd %ulong %57 %58
%60 = OpAccessChain %_ptr_StorageBuffer_ulong %48 %uint_0 %uint_0
%61 = OpLoad %ulong %60
%62 = OpUConvert %uint %61
%63 = OpISub %uint %54 %uint_1
%64 = OpIAdd %uint %63 %62
%65 = OpAccessChain %_ptr_StorageBuffer_ulong %48 %uint_0 %64
%66 = OpLoad %ulong %65
%67 = OpULessThanEqual %bool %59 %66
OpReturnValue %67
OpFunctionEnd
%77 = OpFunction %void None %78
%79 = OpFunctionParameter %uint
%80 = OpFunctionParameter %uint
%81 = OpFunctionParameter %uint
%82 = OpFunctionParameter %uint
%83 = OpLabel
%89 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_0
%92 = OpAtomicIAdd %uint %89 %uint_4 %uint_0 %uint_10
%93 = OpIAdd %uint %92 %uint_10
%94 = OpArrayLength %uint %87 1
%95 = OpULessThanEqual %bool %93 %94
OpSelectionMerge %96 None
OpBranchConditional %95 %97 %96
%97 = OpLabel
%98 = OpIAdd %uint %92 %uint_0
%99 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %98
OpStore %99 %uint_10
%101 = OpIAdd %uint %92 %uint_1
%102 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %101
OpStore %102 %uint_23
%103 = OpIAdd %uint %92 %uint_2
%104 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %103
OpStore %104 %79
%107 = OpIAdd %uint %92 %uint_3
%108 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %107
OpStore %108 %uint_5
%112 = OpLoad %v3uint %gl_GlobalInvocationID
%113 = OpCompositeExtract %uint %112 0
%114 = OpCompositeExtract %uint %112 1
%115 = OpCompositeExtract %uint %112 2
%116 = OpIAdd %uint %92 %uint_4
%117 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %116
OpStore %117 %113
%118 = OpIAdd %uint %92 %uint_5
%119 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %118
OpStore %119 %114
%121 = OpIAdd %uint %92 %uint_6
%122 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %121
OpStore %122 %115
%124 = OpIAdd %uint %92 %uint_7
%125 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %124
OpStore %125 %80
%126 = OpIAdd %uint %92 %uint_8
%127 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %126
OpStore %127 %81
%129 = OpIAdd %uint %92 %uint_9
%130 = OpAccessChain %_ptr_StorageBuffer_uint %87 %uint_1 %129
OpStore %130 %82
OpBranch %96
%96 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBuffAddrCheckPass>(
    defs_before + func_before, defs_after + func_after + new_funcs, true,
    true, 7u, 23u, 2u);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
