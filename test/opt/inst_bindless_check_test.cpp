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
      R"(OpDecorate %_struct_56 Block
OpMemberDecorate %_struct_56 0 Offset 0
OpMemberDecorate %_struct_56 1 Offset 4
OpDecorate %58 DescriptorSet 7
OpDecorate %58 Binding 0
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
%48 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_56 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_56 = OpTypePointer StorageBuffer %_struct_56
%58 = OpVariable %_ptr_StorageBuffer__struct_56 StorageBuffer
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
%106 = OpConstantNull %v4float
)";

  const std::string func_pt1 =
      R"(%MainPs = OpFunction %void None %10
%29 = OpLabel
%30 = OpLoad %v2float %i_vTextureCoords
%31 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%32 = OpLoad %uint %31
%33 = OpAccessChain %_ptr_UniformConstant_16 %g_tColor %32
%34 = OpLoad %16 %33
%35 = OpLoad %24 %g_sAniso
%36 = OpSampledImage %26 %34 %35
)";

  const std::string func_pt2_before =
      R"(%37 = OpImageSampleImplicitLod %v4float %36 %30
OpStore %_entryPointOutput_vColor %37
OpReturn
OpFunctionEnd
)";

  const std::string func_pt2_after =
      R"(%40 = OpULessThan %bool %32 %uint_128
OpSelectionMerge %41 None
OpBranchConditional %40 %42 %43
%42 = OpLabel
%44 = OpLoad %16 %33
%45 = OpSampledImage %26 %44 %35
%46 = OpImageSampleImplicitLod %v4float %45 %30
OpBranch %41
%43 = OpLabel
%105 = OpFunctionCall %void %47 %uint_0 %uint_9 %uint_0 %32 %uint_128
OpBranch %41
%41 = OpLabel
%107 = OpPhi %v4float %46 %42 %106 %43
OpStore %_entryPointOutput_vColor %107
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%47 = OpFunction %void None %48
%49 = OpFunctionParameter %uint
%50 = OpFunctionParameter %uint
%51 = OpFunctionParameter %uint
%52 = OpFunctionParameter %uint
%53 = OpFunctionParameter %uint
%54 = OpLabel
%60 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_0
%63 = OpAtomicIAdd %uint %60 %uint_4 %uint_0 %uint_10
%64 = OpIAdd %uint %63 %uint_10
%65 = OpArrayLength %uint %58 1
%66 = OpULessThanEqual %bool %64 %65
OpSelectionMerge %67 None
OpBranchConditional %66 %68 %67
%68 = OpLabel
%69 = OpIAdd %uint %63 %uint_0
%71 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %69
OpStore %71 %uint_10
%73 = OpIAdd %uint %63 %uint_1
%74 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %73
OpStore %74 %uint_23
%76 = OpIAdd %uint %63 %uint_2
%77 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %76
OpStore %77 %49
%79 = OpIAdd %uint %63 %uint_3
%80 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %79
OpStore %80 %50
%81 = OpIAdd %uint %63 %uint_4
%82 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %81
OpStore %82 %uint_4
%85 = OpLoad %v4float %gl_FragCoord
%87 = OpBitcast %v4uint %85
%88 = OpCompositeExtract %uint %87 0
%90 = OpIAdd %uint %63 %uint_5
%91 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %90
OpStore %91 %88
%92 = OpCompositeExtract %uint %87 1
%94 = OpIAdd %uint %63 %uint_6
%95 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %94
OpStore %95 %92
%97 = OpIAdd %uint %63 %uint_7
%98 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %97
OpStore %98 %51
%100 = OpIAdd %uint %63 %uint_8
%101 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %100
OpStore %101 %52
%103 = OpIAdd %uint %63 %uint_9
%104 = OpAccessChain %_ptr_StorageBuffer_uint %58 %uint_1 %103
OpStore %104 %53
OpBranch %67
%67 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(
    entry_before + names_annots + consts_types_vars + func_pt1 +
    func_pt2_before,
    entry_after + names_annots + new_annots + consts_types_vars +
    new_consts_types_vars + func_pt1 + func_pt2_after + output_func,
    true, true);
}

TEST_F(InstBindlessTest, NoInstrumentConstIndexInbounds) {
  // Texture2D g_tColor[128];
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
  //   ps_output.vColor = g_tColor[ 37 ].Sample(g_sAniso, i.vTextureCoords.xy);
  //   return ps_output;
  // }

  const std::string before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %g_sAniso "g_sAniso"
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_37 = OpConstant %int 37
%15 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_15_uint_128 = OpTypeArray %15 %uint_128
%_ptr_UniformConstant__arr_15_uint_128 = OpTypePointer UniformConstant %_arr_15_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_15_uint_128 UniformConstant
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%21 = OpTypeSampler
%_ptr_UniformConstant_21 = OpTypePointer UniformConstant %21
%g_sAniso = OpVariable %_ptr_UniformConstant_21 UniformConstant
%23 = OpTypeSampledImage %15
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%MainPs = OpFunction %void None %8
%26 = OpLabel
%27 = OpLoad %v2float %i_vTextureCoords
%28 = OpAccessChain %_ptr_UniformConstant_15 %g_tColor %int_37
%29 = OpLoad %15 %28
%30 = OpLoad %21 %g_sAniso
%31 = OpSampledImage %23 %29 %30
%32 = OpImageSampleImplicitLod %v4float %31 %27
OpStore %_entryPointOutput_vColor %32
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(before, before, true, true);
}

TEST_F(InstBindlessTest, InstrumentMultipleInstructions) {
  // Texture2D g_tColor[128];
  //
  // layout(push_constant) cbuffer PerViewConstantBuffer_t
  // {
  //   uint g_nDataIdx;
  //   uint g_nDataIdx2;
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
  //   float t  = g_tColor[g_nDataIdx ].Sample(g_sAniso, i.vTextureCoords.xy);
  //   float t2 = g_tColor[g_nDataIdx2].Sample(g_sAniso, i.vTextureCoords.xy);
  //   ps_output.vColor = t + t2;
  //   return ps_output;
  // }

  const std::string defs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
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
OpMemberDecorate %PerViewConstantBuffer_t 1 Offset 4
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%17 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_17_uint_128 = OpTypeArray %17 %uint_128
%_ptr_UniformConstant__arr_17_uint_128 = OpTypePointer UniformConstant %_arr_17_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_17_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_17 = OpTypePointer UniformConstant %17
%25 = OpTypeSampler
%_ptr_UniformConstant_25 = OpTypePointer UniformConstant %25
%g_sAniso = OpVariable %_ptr_UniformConstant_25 UniformConstant
%27 = OpTypeSampledImage %17
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string defs_after =
      R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
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
OpMemberDecorate %PerViewConstantBuffer_t 1 Offset 4
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
OpDecorate %_struct_64 Block
OpMemberDecorate %_struct_64 0 Offset 0
OpMemberDecorate %_struct_64 1 Offset 4
OpDecorate %66 DescriptorSet 7
OpDecorate %66 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%17 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_17_uint_128 = OpTypeArray %17 %uint_128
%_ptr_UniformConstant__arr_17_uint_128 = OpTypePointer UniformConstant %_arr_17_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_17_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_17 = OpTypePointer UniformConstant %17
%25 = OpTypeSampler
%_ptr_UniformConstant_25 = OpTypePointer UniformConstant %25
%g_sAniso = OpVariable %_ptr_UniformConstant_25 UniformConstant
%27 = OpTypeSampledImage %17
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%56 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_64 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_64 = OpTypePointer StorageBuffer %_struct_64
%66 = OpVariable %_ptr_StorageBuffer__struct_64 StorageBuffer
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
%114 = OpConstantNull %v4float
%uint_15 = OpConstant %uint 15
)";

  const std::string func_before =
      R"(%MainPs = OpFunction %void None %10
%30 = OpLabel
%31 = OpLoad %v2float %i_vTextureCoords
%32 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%33 = OpLoad %uint %32
%34 = OpAccessChain %_ptr_UniformConstant_17 %g_tColor %33
%35 = OpLoad %17 %34
%36 = OpLoad %25 %g_sAniso
%37 = OpSampledImage %27 %35 %36
%38 = OpImageSampleImplicitLod %v4float %37 %31
%39 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
%40 = OpLoad %uint %39
%41 = OpAccessChain %_ptr_UniformConstant_17 %g_tColor %40
%42 = OpLoad %17 %41
%43 = OpSampledImage %27 %42 %36
%44 = OpImageSampleImplicitLod %v4float %43 %31
%45 = OpFAdd %v4float %38 %44
OpStore %_entryPointOutput_vColor %45
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%MainPs = OpFunction %void None %10
%30 = OpLabel
%31 = OpLoad %v2float %i_vTextureCoords
%32 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%33 = OpLoad %uint %32
%34 = OpAccessChain %_ptr_UniformConstant_17 %g_tColor %33
%35 = OpLoad %17 %34
%36 = OpLoad %25 %g_sAniso
%37 = OpSampledImage %27 %35 %36
%48 = OpULessThan %bool %33 %uint_128
OpSelectionMerge %49 None
OpBranchConditional %48 %50 %51
%50 = OpLabel
%52 = OpLoad %17 %34
%53 = OpSampledImage %27 %52 %36
%54 = OpImageSampleImplicitLod %v4float %53 %31
OpBranch %49
%51 = OpLabel
%113 = OpFunctionCall %void %55 %uint_0 %uint_9 %uint_0 %33 %uint_128
OpBranch %49
%49 = OpLabel
%115 = OpPhi %v4float %54 %50 %114 %51
%39 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
%40 = OpLoad %uint %39
%41 = OpAccessChain %_ptr_UniformConstant_17 %g_tColor %40
%42 = OpLoad %17 %41
%43 = OpSampledImage %27 %42 %36
%116 = OpULessThan %bool %40 %uint_128
OpSelectionMerge %117 None
OpBranchConditional %116 %118 %119
%118 = OpLabel
%120 = OpLoad %17 %41
%121 = OpSampledImage %27 %120 %36
%122 = OpImageSampleImplicitLod %v4float %121 %31
OpBranch %117
%119 = OpLabel
%124 = OpFunctionCall %void %55 %uint_0 %uint_15 %uint_0 %40 %uint_128
OpBranch %117
%117 = OpLabel
%125 = OpPhi %v4float %122 %118 %114 %119
%45 = OpFAdd %v4float %115 %125
OpStore %_entryPointOutput_vColor %45
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%55 = OpFunction %void None %56
%57 = OpFunctionParameter %uint
%58 = OpFunctionParameter %uint
%59 = OpFunctionParameter %uint
%60 = OpFunctionParameter %uint
%61 = OpFunctionParameter %uint
%62 = OpLabel
%68 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_0
%71 = OpAtomicIAdd %uint %68 %uint_4 %uint_0 %uint_10
%72 = OpIAdd %uint %71 %uint_10
%73 = OpArrayLength %uint %66 1
%74 = OpULessThanEqual %bool %72 %73
OpSelectionMerge %75 None
OpBranchConditional %74 %76 %75
%76 = OpLabel
%77 = OpIAdd %uint %71 %uint_0
%79 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %77
OpStore %79 %uint_10
%81 = OpIAdd %uint %71 %uint_1
%82 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %81
OpStore %82 %uint_23
%84 = OpIAdd %uint %71 %uint_2
%85 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %84
OpStore %85 %57
%87 = OpIAdd %uint %71 %uint_3
%88 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %87
OpStore %88 %58
%89 = OpIAdd %uint %71 %uint_4
%90 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %89
OpStore %90 %uint_4
%93 = OpLoad %v4float %gl_FragCoord
%95 = OpBitcast %v4uint %93
%96 = OpCompositeExtract %uint %95 0
%98 = OpIAdd %uint %71 %uint_5
%99 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %98
OpStore %99 %96
%100 = OpCompositeExtract %uint %95 1
%102 = OpIAdd %uint %71 %uint_6
%103 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %102
OpStore %103 %100
%105 = OpIAdd %uint %71 %uint_7
%106 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %105
OpStore %106 %59
%108 = OpIAdd %uint %71 %uint_8
%109 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %108
OpStore %109 %60
%111 = OpIAdd %uint %71 %uint_9
%112 = OpAccessChain %_ptr_StorageBuffer_uint %66 %uint_1 %111
OpStore %112 %61
OpBranch %75
%75 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
    defs_after + func_after + output_func, true, true);
}

TEST_F(InstBindlessTest, ReuseConstsTypesBuiltins) {
  // This test verifies that the pass resuses existing constants, types
  // and builtin variables.  This test was created by editing the SPIR-V
  // from the Simple test.

  const std::string defs_before =
    R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
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
OpDecorate %85 DescriptorSet 7
OpDecorate %85 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%20 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_20_uint_128 = OpTypeArray %20 %uint_128
%_ptr_UniformConstant__arr_20_uint_128 = OpTypePointer UniformConstant %_arr_20_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_20_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_20 = OpTypePointer UniformConstant %20
%35 = OpTypeSampler
%_ptr_UniformConstant_35 = OpTypePointer UniformConstant %35
%g_sAniso = OpVariable %_ptr_UniformConstant_35 UniformConstant
%39 = OpTypeSampledImage %20
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_83 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_83 = OpTypePointer StorageBuffer %_struct_83
%85 = OpVariable %_ptr_StorageBuffer__struct_83 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_4 = OpConstant %uint 4
%uint_1 = OpConstant %uint 1
%uint_23 = OpConstant %uint 23
%uint_2 = OpConstant %uint 2
%uint_9 = OpConstant %uint 9
%uint_3 = OpConstant %uint 3
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%v4uint = OpTypeVector %uint 4
%uint_5 = OpConstant %uint 5
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%131 = OpConstantNull %v4float
)";

  const std::string defs_after =
    R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
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
OpDecorate %10 DescriptorSet 7
OpDecorate %10 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
OpDecorate %_struct_34 Block
OpMemberDecorate %_struct_34 0 Offset 0
OpMemberDecorate %_struct_34 1 Offset 4
OpDecorate %75 DescriptorSet 7
OpDecorate %75 Binding 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%18 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%_arr_18_uint_128 = OpTypeArray %18 %uint_128
%_ptr_UniformConstant__arr_18_uint_128 = OpTypePointer UniformConstant %_arr_18_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_18_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%26 = OpTypeSampler
%_ptr_UniformConstant_26 = OpTypePointer UniformConstant %26
%g_sAniso = OpVariable %_ptr_UniformConstant_26 UniformConstant
%28 = OpTypeSampledImage %18
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_34 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_34 = OpTypePointer StorageBuffer %_struct_34
%10 = OpVariable %_ptr_StorageBuffer__struct_34 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_4 = OpConstant %uint 4
%uint_1 = OpConstant %uint 1
%uint_23 = OpConstant %uint 23
%uint_2 = OpConstant %uint 2
%uint_9 = OpConstant %uint 9
%uint_3 = OpConstant %uint 3
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%v4uint = OpTypeVector %uint 4
%uint_5 = OpConstant %uint 5
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%50 = OpConstantNull %v4float
%68 = OpTypeFunction %void %uint %uint %uint %uint %uint
%75 = OpVariable %_ptr_StorageBuffer__struct_34 StorageBuffer
)";

  const std::string func_before =
    R"(%MainPs = OpFunction %void None %3
%5 = OpLabel
%53 = OpLoad %v2float %i_vTextureCoords
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%64 = OpLoad %uint %63
%65 = OpAccessChain %_ptr_UniformConstant_20 %g_tColor %64
%67 = OpLoad %35 %g_sAniso
%78 = OpLoad %20 %65
%79 = OpSampledImage %39 %78 %67
%71 = OpImageSampleImplicitLod %v4float %79 %53
OpStore %_entryPointOutput_vColor %71
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
    R"(%MainPs = OpFunction %void None %12
%51 = OpLabel
%52 = OpLoad %v2float %i_vTextureCoords
%53 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%54 = OpLoad %uint %53
%55 = OpAccessChain %_ptr_UniformConstant_18 %g_tColor %54
%56 = OpLoad %26 %g_sAniso
%57 = OpLoad %18 %55
%58 = OpSampledImage %28 %57 %56
%60 = OpULessThan %bool %54 %uint_128
OpSelectionMerge %61 None
OpBranchConditional %60 %62 %63
%62 = OpLabel
%64 = OpLoad %18 %55
%65 = OpSampledImage %28 %64 %56
%66 = OpImageSampleImplicitLod %v4float %65 %52
OpBranch %61
%63 = OpLabel
%107 = OpFunctionCall %void %67 %uint_0 %uint_9 %uint_0 %54 %uint_128
OpBranch %61
%61 = OpLabel
%108 = OpPhi %v4float %66 %62 %50 %63
OpStore %_entryPointOutput_vColor %108
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
    R"(%67 = OpFunction %void None %68
%69 = OpFunctionParameter %uint
%70 = OpFunctionParameter %uint
%71 = OpFunctionParameter %uint
%72 = OpFunctionParameter %uint
%73 = OpFunctionParameter %uint
%74 = OpLabel
%76 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_0
%77 = OpAtomicIAdd %uint %76 %uint_4 %uint_0 %uint_10
%78 = OpIAdd %uint %77 %uint_10
%79 = OpArrayLength %uint %75 1
%80 = OpULessThanEqual %bool %78 %79
OpSelectionMerge %81 None
OpBranchConditional %80 %82 %81
%82 = OpLabel
%83 = OpIAdd %uint %77 %uint_0
%84 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %83
OpStore %84 %uint_10
%85 = OpIAdd %uint %77 %uint_1
%86 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %85
OpStore %86 %uint_23
%87 = OpIAdd %uint %77 %uint_2
%88 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %87
OpStore %88 %69
%89 = OpIAdd %uint %77 %uint_3
%90 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %89
OpStore %90 %70
%91 = OpIAdd %uint %77 %uint_4
%92 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %91
OpStore %92 %uint_4
%93 = OpLoad %v4float %gl_FragCoord
%94 = OpBitcast %v4uint %93
%95 = OpCompositeExtract %uint %94 0
%96 = OpIAdd %uint %77 %uint_5
%97 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %96
OpStore %97 %95
%98 = OpCompositeExtract %uint %94 1
%99 = OpIAdd %uint %77 %uint_6
%100 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %99
OpStore %100 %98
%101 = OpIAdd %uint %77 %uint_7
%102 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %101
OpStore %102 %71
%103 = OpIAdd %uint %77 %uint_8
%104 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %103
OpStore %104 %72
%105 = OpIAdd %uint %77 %uint_9
%106 = OpAccessChain %_ptr_StorageBuffer_uint %75 %uint_1 %105
OpStore %106 %73
OpBranch %81
%81 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
    defs_after + func_after + output_func, true, true);
}

TEST_F(InstBindlessTest, InstrumentOpImage) {
  // This test verifies that the pass will correctly instrument shader
  // using OpImage. This test was created by editing the SPIR-V
  // from the Simple test.

  const std::string defs_before =
      R"(OpCapability Shader
OpCapability StorageImageReadWithoutFormat
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%int_0 = OpConstant %int 0
%20 = OpTypeImage %float 2D 0 0 0 0 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%39 = OpTypeSampledImage %20
%_arr_39_uint_128 = OpTypeArray %39 %uint_128
%_ptr_UniformConstant__arr_39_uint_128 = OpTypePointer UniformConstant %_arr_39_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_39_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_39 = OpTypePointer UniformConstant %39
%_ptr_Input_v2int = OpTypePointer Input %v2int
%i_vTextureCoords = OpVariable %_ptr_Input_v2int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string defs_after =
      R"(OpCapability Shader
OpCapability StorageImageReadWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
OpDecorate %_struct_52 Block
OpMemberDecorate %_struct_52 0 Offset 0
OpMemberDecorate %_struct_52 1 Offset 4
OpDecorate %54 DescriptorSet 7
OpDecorate %54 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%int_0 = OpConstant %int 0
%15 = OpTypeImage %float 2D 0 0 0 0 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%18 = OpTypeSampledImage %15
%_arr_18_uint_128 = OpTypeArray %18 %uint_128
%_ptr_UniformConstant__arr_18_uint_128 = OpTypePointer UniformConstant %_arr_18_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_18_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%_ptr_Input_v2int = OpTypePointer Input %v2int
%i_vTextureCoords = OpVariable %_ptr_Input_v2int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%44 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_52 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_52 = OpTypePointer StorageBuffer %_struct_52
%54 = OpVariable %_ptr_StorageBuffer__struct_52 StorageBuffer
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
%102 = OpConstantNull %v4float
)";

  const std::string func_before =
      R"(%MainPs = OpFunction %void None %3
%5 = OpLabel
%53 = OpLoad %v2int %i_vTextureCoords
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%64 = OpLoad %uint %63
%65 = OpAccessChain %_ptr_UniformConstant_39 %g_tColor %64
%66 = OpLoad %39 %65
%75 = OpImage %20 %66
%71 = OpImageRead %v4float %75 %53
OpStore %_entryPointOutput_vColor %71
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%MainPs = OpFunction %void None %9
%26 = OpLabel
%27 = OpLoad %v2int %i_vTextureCoords
%28 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%29 = OpLoad %uint %28
%30 = OpAccessChain %_ptr_UniformConstant_18 %g_tColor %29
%31 = OpLoad %18 %30
%32 = OpImage %15 %31
%36 = OpULessThan %bool %29 %uint_128
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %39
%38 = OpLabel
%40 = OpLoad %18 %30
%41 = OpImage %15 %40
%42 = OpImageRead %v4float %41 %27
OpBranch %37
%39 = OpLabel
%101 = OpFunctionCall %void %43 %uint_0 %uint_8 %uint_0 %29 %uint_128
OpBranch %37
%37 = OpLabel
%103 = OpPhi %v4float %42 %38 %102 %39
OpStore %_entryPointOutput_vColor %103
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%43 = OpFunction %void None %44
%45 = OpFunctionParameter %uint
%46 = OpFunctionParameter %uint
%47 = OpFunctionParameter %uint
%48 = OpFunctionParameter %uint
%49 = OpFunctionParameter %uint
%50 = OpLabel
%56 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_0
%59 = OpAtomicIAdd %uint %56 %uint_4 %uint_0 %uint_10
%60 = OpIAdd %uint %59 %uint_10
%61 = OpArrayLength %uint %54 1
%62 = OpULessThanEqual %bool %60 %61
OpSelectionMerge %63 None
OpBranchConditional %62 %64 %63
%64 = OpLabel
%65 = OpIAdd %uint %59 %uint_0
%67 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %65
OpStore %67 %uint_10
%69 = OpIAdd %uint %59 %uint_1
%70 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %69
OpStore %70 %uint_23
%72 = OpIAdd %uint %59 %uint_2
%73 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %72
OpStore %73 %45
%75 = OpIAdd %uint %59 %uint_3
%76 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %75
OpStore %76 %46
%77 = OpIAdd %uint %59 %uint_4
%78 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %77
OpStore %78 %uint_4
%81 = OpLoad %v4float %gl_FragCoord
%83 = OpBitcast %v4uint %81
%84 = OpCompositeExtract %uint %83 0
%86 = OpIAdd %uint %59 %uint_5
%87 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %86
OpStore %87 %84
%88 = OpCompositeExtract %uint %83 1
%90 = OpIAdd %uint %59 %uint_6
%91 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %90
OpStore %91 %88
%93 = OpIAdd %uint %59 %uint_7
%94 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %93
OpStore %94 %47
%96 = OpIAdd %uint %59 %uint_8
%97 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %96
OpStore %97 %48
%99 = OpIAdd %uint %59 %uint_9
%100 = OpAccessChain %_ptr_StorageBuffer_uint %54 %uint_1 %99
OpStore %100 %49
OpBranch %63
%63 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
      defs_after + func_after + output_func, true, true);
}

TEST_F(InstBindlessTest, InstrumentSampledImage) {
  // This test verifies that the pass will correctly instrument shader
  // using sampled image. This test was created by editing the SPIR-V
  // from the Simple test.

  const std::string defs_before =
    R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%20 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%39 = OpTypeSampledImage %20
%_arr_39_uint_128 = OpTypeArray %39 %uint_128
%_ptr_UniformConstant__arr_39_uint_128 = OpTypePointer UniformConstant %_arr_39_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_39_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_39 = OpTypePointer UniformConstant %39
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string defs_after =
    R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
OpDecorate %_struct_50 Block
OpMemberDecorate %_struct_50 0 Offset 0
OpMemberDecorate %_struct_50 1 Offset 4
OpDecorate %52 DescriptorSet 7
OpDecorate %52 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%15 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%18 = OpTypeSampledImage %15
%_arr_18_uint_128 = OpTypeArray %18 %uint_128
%_ptr_UniformConstant__arr_18_uint_128 = OpTypePointer UniformConstant %_arr_18_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_18_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%42 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_50 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_50 = OpTypePointer StorageBuffer %_struct_50
%52 = OpVariable %_ptr_StorageBuffer__struct_50 StorageBuffer
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
%100 = OpConstantNull %v4float
)";

  const std::string func_before =
    R"(%MainPs = OpFunction %void None %3
%5 = OpLabel
%53 = OpLoad %v2float %i_vTextureCoords
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%64 = OpLoad %uint %63
%65 = OpAccessChain %_ptr_UniformConstant_39 %g_tColor %64
%66 = OpLoad %39 %65
%71 = OpImageSampleImplicitLod %v4float %66 %53
OpStore %_entryPointOutput_vColor %71
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
    R"(%MainPs = OpFunction %void None %9
%26 = OpLabel
%27 = OpLoad %v2float %i_vTextureCoords
%28 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%29 = OpLoad %uint %28
%30 = OpAccessChain %_ptr_UniformConstant_18 %g_tColor %29
%31 = OpLoad %18 %30
%35 = OpULessThan %bool %29 %uint_128
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %38
%37 = OpLabel
%39 = OpLoad %18 %30
%40 = OpImageSampleImplicitLod %v4float %39 %27
OpBranch %36
%38 = OpLabel
%99 = OpFunctionCall %void %41 %uint_0 %uint_7 %uint_0 %29 %uint_128
OpBranch %36
%36 = OpLabel
%101 = OpPhi %v4float %40 %37 %100 %38
OpStore %_entryPointOutput_vColor %101
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
    R"(%41 = OpFunction %void None %42
%43 = OpFunctionParameter %uint
%44 = OpFunctionParameter %uint
%45 = OpFunctionParameter %uint
%46 = OpFunctionParameter %uint
%47 = OpFunctionParameter %uint
%48 = OpLabel
%54 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_0
%57 = OpAtomicIAdd %uint %54 %uint_4 %uint_0 %uint_10
%58 = OpIAdd %uint %57 %uint_10
%59 = OpArrayLength %uint %52 1
%60 = OpULessThanEqual %bool %58 %59
OpSelectionMerge %61 None
OpBranchConditional %60 %62 %61
%62 = OpLabel
%63 = OpIAdd %uint %57 %uint_0
%65 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %63
OpStore %65 %uint_10
%67 = OpIAdd %uint %57 %uint_1
%68 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %67
OpStore %68 %uint_23
%70 = OpIAdd %uint %57 %uint_2
%71 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %70
OpStore %71 %43
%73 = OpIAdd %uint %57 %uint_3
%74 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %73
OpStore %74 %44
%75 = OpIAdd %uint %57 %uint_4
%76 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %75
OpStore %76 %uint_4
%79 = OpLoad %v4float %gl_FragCoord
%81 = OpBitcast %v4uint %79
%82 = OpCompositeExtract %uint %81 0
%84 = OpIAdd %uint %57 %uint_5
%85 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %84
OpStore %85 %82
%86 = OpCompositeExtract %uint %81 1
%88 = OpIAdd %uint %57 %uint_6
%89 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %88
OpStore %89 %86
%91 = OpIAdd %uint %57 %uint_7
%92 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %91
OpStore %92 %45
%94 = OpIAdd %uint %57 %uint_8
%95 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %94
OpStore %95 %46
%97 = OpIAdd %uint %57 %uint_9
%98 = OpAccessChain %_ptr_StorageBuffer_uint %52 %uint_1 %97
OpStore %98 %47
OpBranch %61
%61 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
    defs_after + func_after + output_func, true, true);
}

TEST_F(InstBindlessTest, InstrumentImageWrite) {
  // This test verifies that the pass will correctly instrument shader
  // doing bindless image write. This test was created by editing the SPIR-V
  // from the Simple test.

  const std::string defs_before =
    R"(OpCapability Shader
OpCapability StorageImageWriteWithoutFormat
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%int_0 = OpConstant %int 0
%20 = OpTypeImage %float 2D 0 0 0 0 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%80 = OpConstantNull %v4float
%_arr_20_uint_128 = OpTypeArray %20 %uint_128
%_ptr_UniformConstant__arr_20_uint_128 = OpTypePointer UniformConstant %_arr_20_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_20_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_20 = OpTypePointer UniformConstant %20
%_ptr_Input_v2int = OpTypePointer Input %v2int
%i_vTextureCoords = OpVariable %_ptr_Input_v2int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string defs_after =
    R"(OpCapability Shader
OpCapability StorageImageWriteWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor %gl_FragCoord
OpExecutionMode %MainPs OriginUpperLeft
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %g_tColor "g_tColor"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpName %_ ""
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpDecorate %g_tColor DescriptorSet 3
OpDecorate %g_tColor Binding 0
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
OpDecorate %_struct_49 Block
OpMemberDecorate %_struct_49 0 Offset 0
OpMemberDecorate %_struct_49 1 Offset 4
OpDecorate %51 DescriptorSet 7
OpDecorate %51 Binding 0
OpDecorate %gl_FragCoord BuiltIn FragCoord
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%int_0 = OpConstant %int 0
%16 = OpTypeImage %float 2D 0 0 0 0 Unknown
%uint = OpTypeInt 32 0
%uint_128 = OpConstant %uint 128
%19 = OpConstantNull %v4float
%_arr_16_uint_128 = OpTypeArray %16 %uint_128
%_ptr_UniformConstant__arr_16_uint_128 = OpTypePointer UniformConstant %_arr_16_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_16_uint_128 UniformConstant
%PerViewConstantBuffer_t = OpTypeStruct %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%_ptr_UniformConstant_16 = OpTypePointer UniformConstant %16
%_ptr_Input_v2int = OpTypePointer Input %v2int
%i_vTextureCoords = OpVariable %_ptr_Input_v2int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%41 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_49 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_49 = OpTypePointer StorageBuffer %_struct_49
%51 = OpVariable %_ptr_StorageBuffer__struct_49 StorageBuffer
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
)";

  const std::string func_before =
    R"(%MainPs = OpFunction %void None %3
%5 = OpLabel
%53 = OpLoad %v2int %i_vTextureCoords
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%64 = OpLoad %uint %63
%65 = OpAccessChain %_ptr_UniformConstant_20 %g_tColor %64
%66 = OpLoad %20 %65
OpImageWrite %66 %53 %80
OpStore %_entryPointOutput_vColor %80
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
    R"(%MainPs = OpFunction %void None %9
%27 = OpLabel
%28 = OpLoad %v2int %i_vTextureCoords
%29 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%30 = OpLoad %uint %29
%31 = OpAccessChain %_ptr_UniformConstant_16 %g_tColor %30
%32 = OpLoad %16 %31
%35 = OpULessThan %bool %30 %uint_128
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %38
%37 = OpLabel
%39 = OpLoad %16 %31
OpImageWrite %39 %28 %19
OpBranch %36
%38 = OpLabel
%98 = OpFunctionCall %void %40 %uint_0 %uint_7 %uint_0 %30 %uint_128
OpBranch %36
%36 = OpLabel
OpStore %_entryPointOutput_vColor %19
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
    R"(%40 = OpFunction %void None %41
%42 = OpFunctionParameter %uint
%43 = OpFunctionParameter %uint
%44 = OpFunctionParameter %uint
%45 = OpFunctionParameter %uint
%46 = OpFunctionParameter %uint
%47 = OpLabel
%53 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_0
%56 = OpAtomicIAdd %uint %53 %uint_4 %uint_0 %uint_10
%57 = OpIAdd %uint %56 %uint_10
%58 = OpArrayLength %uint %51 1
%59 = OpULessThanEqual %bool %57 %58
OpSelectionMerge %60 None
OpBranchConditional %59 %61 %60
%61 = OpLabel
%62 = OpIAdd %uint %56 %uint_0
%64 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %62
OpStore %64 %uint_10
%66 = OpIAdd %uint %56 %uint_1
%67 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %66
OpStore %67 %uint_23
%69 = OpIAdd %uint %56 %uint_2
%70 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %69
OpStore %70 %42
%72 = OpIAdd %uint %56 %uint_3
%73 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %72
OpStore %73 %43
%74 = OpIAdd %uint %56 %uint_4
%75 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %74
OpStore %75 %uint_4
%78 = OpLoad %v4float %gl_FragCoord
%80 = OpBitcast %v4uint %78
%81 = OpCompositeExtract %uint %80 0
%83 = OpIAdd %uint %56 %uint_5
%84 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %83
OpStore %84 %81
%85 = OpCompositeExtract %uint %80 1
%87 = OpIAdd %uint %56 %uint_6
%88 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %87
OpStore %88 %85
%90 = OpIAdd %uint %56 %uint_7
%91 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %90
OpStore %91 %44
%93 = OpIAdd %uint %56 %uint_8
%94 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %93
OpStore %94 %45
%96 = OpIAdd %uint %56 %uint_9
%97 = OpAccessChain %_ptr_StorageBuffer_uint %51 %uint_1 %96
OpStore %97 %46
OpBranch %60
%60 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
    defs_after + func_after + output_func, true, true);
}

TEST_F(InstBindlessTest, InstrumentVertexSimple) {
  // This test verifies that the pass will correctly instrument shader
  // doing bindless image write. This test was created by editing the SPIR-V
  // from the Simple test.

  const std::string defs_before =
      R"(OpCapability Shader
OpCapability Sampled1D
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %_ %coords2D
OpSource GLSL 450
OpName %main "main"
OpName %lod "lod"
OpName %coords1D "coords1D"
OpName %gl_PerVertex "gl_PerVertex"
OpMemberName %gl_PerVertex 0 "gl_Position"
OpMemberName %gl_PerVertex 1 "gl_PointSize"
OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
OpMemberName %gl_PerVertex 3 "gl_CullDistance"
OpName %_ ""
OpName %texSampler1D "texSampler1D"
OpName %foo "foo"
OpMemberName %foo 0 "g_idx"
OpName %__0 ""
OpName %coords2D "coords2D"
OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
OpDecorate %gl_PerVertex Block
OpDecorate %texSampler1D DescriptorSet 0
OpDecorate %texSampler1D Binding 3
OpMemberDecorate %foo 0 Offset 0
OpDecorate %foo Block
OpDecorate %__0 DescriptorSet 0
OpDecorate %__0 Binding 5
OpDecorate %coords2D Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_3 = OpConstant %float 3
%float_1_78900003 = OpConstant %float 1.78900003
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_PerVertex = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
%_ = OpVariable %_ptr_Output_gl_PerVertex Output
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%21 = OpTypeImage %float 1D 0 0 0 1 Unknown
%22 = OpTypeSampledImage %21
%uint_128 = OpConstant %uint 128
%_arr_22_uint_128 = OpTypeArray %22 %uint_128
%_ptr_UniformConstant__arr_22_uint_128 = OpTypePointer UniformConstant %_arr_22_uint_128
%texSampler1D = OpVariable %_ptr_UniformConstant__arr_22_uint_128 UniformConstant
%foo = OpTypeStruct %int
%_ptr_Uniform_foo = OpTypePointer Uniform %foo
%__0 = OpVariable %_ptr_Uniform_foo Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
%_ptr_UniformConstant_22 = OpTypePointer UniformConstant %22
%_ptr_Output_v4float = OpTypePointer Output %v4float
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%coords2D = OpVariable %_ptr_Input_v2float Input
)";

  const std::string defs_after =
      R"(OpCapability Shader
OpCapability Sampled1D
OpExtension "SPV_KHR_storage_buffer_storage_class"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %_ %coords2D %gl_VertexID %gl_InstanceID
OpSource GLSL 450
OpName %main "main"
OpName %lod "lod"
OpName %coords1D "coords1D"
OpName %gl_PerVertex "gl_PerVertex"
OpMemberName %gl_PerVertex 0 "gl_Position"
OpMemberName %gl_PerVertex 1 "gl_PointSize"
OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
OpMemberName %gl_PerVertex 3 "gl_CullDistance"
OpName %_ ""
OpName %texSampler1D "texSampler1D"
OpName %foo "foo"
OpMemberName %foo 0 "g_idx"
OpName %__0 ""
OpName %coords2D "coords2D"
OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
OpDecorate %gl_PerVertex Block
OpDecorate %texSampler1D DescriptorSet 0
OpDecorate %texSampler1D Binding 3
OpMemberDecorate %foo 0 Offset 0
OpDecorate %foo Block
OpDecorate %__0 DescriptorSet 0
OpDecorate %__0 Binding 5
OpDecorate %coords2D Location 0
OpDecorate %_struct_62 Block
OpMemberDecorate %_struct_62 0 Offset 0
OpMemberDecorate %_struct_62 1 Offset 4
OpDecorate %64 DescriptorSet 7
OpDecorate %64 Binding 0
OpDecorate %gl_VertexID BuiltIn VertexId
OpDecorate %gl_InstanceID BuiltIn InstanceId
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_3 = OpConstant %float 3
%float_1_78900003 = OpConstant %float 1.78900003
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_PerVertex = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
%_ = OpVariable %_ptr_Output_gl_PerVertex Output
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%24 = OpTypeImage %float 1D 0 0 0 1 Unknown
%25 = OpTypeSampledImage %24
%uint_128 = OpConstant %uint 128
%_arr_25_uint_128 = OpTypeArray %25 %uint_128
%_ptr_UniformConstant__arr_25_uint_128 = OpTypePointer UniformConstant %_arr_25_uint_128
%texSampler1D = OpVariable %_ptr_UniformConstant__arr_25_uint_128 UniformConstant
%foo = OpTypeStruct %int
%_ptr_Uniform_foo = OpTypePointer Uniform %foo
%__0 = OpVariable %_ptr_Uniform_foo Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
%_ptr_UniformConstant_25 = OpTypePointer UniformConstant %25
%_ptr_Output_v4float = OpTypePointer Output %v4float
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%coords2D = OpVariable %_ptr_Input_v2float Input
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%54 = OpTypeFunction %void %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_62 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_62 = OpTypePointer StorageBuffer %_struct_62
%64 = OpVariable %_ptr_StorageBuffer__struct_62 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_10 = OpConstant %uint 10
%uint_4 = OpConstant %uint 4
%uint_23 = OpConstant %uint 23
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%_ptr_Input_uint = OpTypePointer Input %uint
%gl_VertexID = OpVariable %_ptr_Input_uint Input
%uint_5 = OpConstant %uint 5
%gl_InstanceID = OpVariable %_ptr_Input_uint Input
%uint_6 = OpConstant %uint 6
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%uint_9 = OpConstant %uint 9
%uint_12 = OpConstant %uint 12
%110 = OpConstantNull %v4float
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %3
%5 = OpLabel
%lod = OpVariable %_ptr_Function_float Function
%coords1D = OpVariable %_ptr_Function_float Function
OpStore %lod %float_3
OpStore %coords1D %float_1_78900003
%31 = OpAccessChain %_ptr_Uniform_int %__0 %int_0
%32 = OpLoad %int %31
%34 = OpAccessChain %_ptr_UniformConstant_22 %texSampler1D %32
%35 = OpLoad %22 %34
%36 = OpLoad %float %coords1D
%37 = OpLoad %float %lod
%38 = OpImageSampleExplicitLod %v4float %35 %36 Lod %37
%40 = OpAccessChain %_ptr_Output_v4float %_ %int_0
OpStore %40 %38
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %12
%35 = OpLabel
%lod = OpVariable %_ptr_Function_float Function
%coords1D = OpVariable %_ptr_Function_float Function
OpStore %lod %float_3
OpStore %coords1D %float_1_78900003
%36 = OpAccessChain %_ptr_Uniform_int %__0 %int_0
%37 = OpLoad %int %36
%38 = OpAccessChain %_ptr_UniformConstant_25 %texSampler1D %37
%39 = OpLoad %25 %38
%40 = OpLoad %float %coords1D
%41 = OpLoad %float %lod
%46 = OpULessThan %bool %37 %uint_128
OpSelectionMerge %47 None
OpBranchConditional %46 %48 %49
%48 = OpLabel
%50 = OpLoad %25 %38
%51 = OpImageSampleExplicitLod %v4float %50 %40 Lod %41
OpBranch %47
%49 = OpLabel
%52 = OpBitcast %uint %37
%109 = OpFunctionCall %void %53 %uint_0 %uint_12 %uint_0 %52 %uint_128
OpBranch %47
%47 = OpLabel
%111 = OpPhi %v4float %51 %48 %110 %49
%43 = OpAccessChain %_ptr_Output_v4float %_ %int_0
OpStore %43 %111
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%53 = OpFunction %void None %54
%55 = OpFunctionParameter %uint
%56 = OpFunctionParameter %uint
%57 = OpFunctionParameter %uint
%58 = OpFunctionParameter %uint
%59 = OpFunctionParameter %uint
%60 = OpLabel
%66 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_0
%69 = OpAtomicIAdd %uint %66 %uint_4 %uint_0 %uint_10
%70 = OpIAdd %uint %69 %uint_10
%71 = OpArrayLength %uint %64 1
%72 = OpULessThanEqual %bool %70 %71
OpSelectionMerge %73 None
OpBranchConditional %72 %74 %73
%74 = OpLabel
%75 = OpIAdd %uint %69 %uint_0
%76 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %75
OpStore %76 %uint_10
%78 = OpIAdd %uint %69 %uint_1
%79 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %78
OpStore %79 %uint_23
%81 = OpIAdd %uint %69 %uint_2
%82 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %81
OpStore %82 %55
%84 = OpIAdd %uint %69 %uint_3
%85 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %84
OpStore %85 %56
%86 = OpIAdd %uint %69 %uint_4
%87 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %86
OpStore %87 %uint_0
%90 = OpLoad %uint %gl_VertexID
%92 = OpIAdd %uint %69 %uint_5
%93 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %92
OpStore %93 %90
%95 = OpLoad %uint %gl_InstanceID
%97 = OpIAdd %uint %69 %uint_6
%98 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %97
OpStore %98 %95
%100 = OpIAdd %uint %69 %uint_7
%101 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %100
OpStore %101 %57
%103 = OpIAdd %uint %69 %uint_8
%104 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %103
OpStore %104 %58
%106 = OpIAdd %uint %69 %uint_9
%107 = OpAccessChain %_ptr_StorageBuffer_uint %64 %uint_1 %106
OpStore %107 %59
OpBranch %73
%73 = OpLabel
OpReturn
OpFunctionEnd
)";

  // SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstBindlessCheckPass>(defs_before + func_before,
    defs_after + func_after + output_func, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
// TODO(greg-lunarg): Come up with cases to put here :)

}  // namespace
}  // namespace opt
}  // namespace spvtools
