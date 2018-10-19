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

#include <string>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InstBindlessTest = PassTest<::testing::Test>;

TEST_F(InstBindlessTest, Simple) {
  // Texture2D g_tColor[128];
  //
  // layout(push_constant) cbuffer PerViewConstantBuffer_t
  // {
  //   uint g_nDataIdx;
  // };
  //
  // SamplerState g_sAniso;
  //
  // struct PS_INPUT
  // {
  //   float2 vTextureCoords : TEXCOORD2;
  // };
  //
  // struct PS_OUTPUT
  // {
  //   float4 vColor : SV_Target0;
  // };
  //
  // PS_OUTPUT MainPs(PS_INPUT i)
  // {
  //   PS_OUTPUT ps_output;
  //
  //   ps_output.vColor = g_tColor[ g_nDataIdx ].Sample(g_sAniso, i.vTextureCoords.xy);
  //   return ps_output;
  // }


  const std::string entry_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
)";

  const std::string entry_after =
      R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
)";

  const std::string names_annots =
      R"(OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %g_sAniso "g_sAniso"
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
)";

  const std::string new_annots =
      R"(OpDecorate %_struct_57 Block
OpMemberDecorate %_struct_57 0 Offset 0
OpMemberDecorate %_struct_57 1 Offset 4
OpDecorate %59 DescriptorSet 7
OpDecorate %59 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
)";

  const std::string consts_types_vars =
    R"(%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%16 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_16_uint_128 = OpTypeArray %16 %uint_128
%_ptr_UniformConstant__arr_16_uint_128 = OpTypePointer UniformConstant %_arr_16_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_16_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_16 = OpTypePointer UniformConstant %16
%24 = OpTypeSampler
%_ptr_UniformConstant_24 = OpTypePointer UniformConstant %24
%g_sAniso = OpVariable %_ptr_UniformConstant_24 UniformConstant
%26 = OpTypeSampledImage %16
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string new_consts_types_vars =
      R"(%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%49 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_57 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_57 = OpTypePointer StorageBuffer %_struct_57
%59 = OpVariable %_ptr_StorageBuffer__struct_57 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_4 = OpConstant %uint 4
%uint_1 = OpConstant %uint 1
%uint_23 = OpConstant %uint 23
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%v4uint = OpTypeVector %uint 4
%uint_5 = OpConstant %uint 5
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%uint_9 = OpConstant %uint 9
%107 = OpConstantNull %v4float
)";

  const std::string func_before =
      R"(%MainPs = OpFunction %void None %10
%29 = OpLabel
%30 = OpLoad %v2float %i_vTextureCoords
%31 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%32 = OpLoad %uint %31
%33 = OpAccessChain %_ptr_UniformConstant_16 %g_tColor %32
%34 = OpLoad %16 %33
%35 = OpLoad %24 %g_sAniso
%36 = OpSampledImage %26 %34 %35
%37 = OpImageSampleImplicitLod %v4float %36 %30
OpStore %_entryPointOutput_vColor %37
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%MainPs = OpFunction %void None %10
%38 = OpLabel
%30 = OpLoad %v2float %i_vTextureCoords
%31 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%32 = OpLoad %uint %31
%33 = OpAccessChain %_ptr_UniformConstant_16 %g_tColor %32
%34 = OpLoad %16 %33
%35 = OpLoad %24 %g_sAniso
%36 = OpSampledImage %26 %34 %35
%41 = OpULessThan %bool %32 %uint_128
OpSelectionMerge %42 None
OpBranchConditional %41 %43 %44
%43 = OpLabel
%45 = OpLoad %16 %33
%46 = OpSampledImage %26 %45 %35
%47 = OpImageSampleImplicitLod %v4float %46 %30
OpBranch %42
%44 = OpLabel
%106 = OpFunctionCall %void %48 %uint_0 %uint_9 %uint_0 %32 %uint_128
OpBranch %42
%42 = OpLabel
%108 = OpPhi %v4float %47 %43 %107 %44
OpStore %_entryPointOutput_vColor %108
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%48 = OpFunction %void None %49
%50 = OpFunctionParameter %uint
%51 = OpFunctionParameter %uint
%52 = OpFunctionParameter %uint
%53 = OpFunctionParameter %uint
%54 = OpFunctionParameter %uint
%55 = OpLabel
%61 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_0
%64 = OpAtomicIAdd %uint %61 %uint_4 %uint_0 %uint_10
%65 = OpIAdd %uint %64 %uint_10
%66 = OpArrayLength %uint %59 1
%67 = OpULessThanEqual %bool %65 %66
OpSelectionMerge %68 None
OpBranchConditional %67 %69 %68
%69 = OpLabel
%70 = OpIAdd %uint %64 %uint_0
%72 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %70
OpStore %72 %uint_10
%74 = OpIAdd %uint %64 %uint_1
%75 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %74
OpStore %75 %uint_23
%77 = OpIAdd %uint %64 %uint_2
%78 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %77
OpStore %78 %50
%80 = OpIAdd %uint %64 %uint_3
%81 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %80
OpStore %81 %51
%82 = OpIAdd %uint %64 %uint_4
%83 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %82
OpStore %83 %uint_4
%86 = OpLoad %v4float %gl_FragCoord
%88 = OpBitcast %v4uint %86
%89 = OpCompositeExtract %uint %88 0
%91 = OpIAdd %uint %64 %uint_5
%92 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %91
OpStore %92 %89
%93 = OpCompositeExtract %uint %88 1
%95 = OpIAdd %uint %64 %uint_6
%96 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %95
OpStore %96 %93
%98 = OpIAdd %uint %64 %uint_7
%99 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %98
OpStore %99 %52
%101 = OpIAdd %uint %64 %uint_8
%102 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %101
OpStore %102 %53
%104 = OpIAdd %uint %64 %uint_9
%105 = OpAccessChain %_ptr_StorageBuffer_uint %59 %uint_1 %104
OpStore %105 %54
OpBranch %68
%68 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(
      entry_before + names_annots + consts_types_vars + func_before,
      entry_after + names_annots + new_annots + consts_types_vars + new_consts_types_vars + func_after + output_func,
      true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
// TODO(greg-lunarg): Come up with stuff to put here :)

}  // namespace
}  // namespace opt
}  // namespace spvtools
