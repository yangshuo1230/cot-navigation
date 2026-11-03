# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/ev/cev3th4appgglnhifdzu7nnaxg42hzydo5a3ryj6eob2g7rhm363.py
# Topologically Sorted Source Nodes: [x, layer_norm, linear], Original ATen: [aten.embedding, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm => add_11, add_12, mul_8, mul_9, rsqrt, sub_8, var_mean
#   linear => convert_element_type_1
#   x => embedding
# Graph fragment:
#   %embedding : [num_users=4] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %primals_3, 0), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%embedding, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %getitem_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_5), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_6), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused__to_copy_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_embedding_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_embedding_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, RBLOCK], 270, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 270)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 270")
    tmp6 = tl.load(in_ptr1 + (r1 + 128*tmp4), xmask, other=0.0)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 128.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp6 - tmp16
    tmp29 = tmp28 * tmp27
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr0 + (r1 + 128*x0), tmp6, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp34, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/eo/ceostb5wf7nadfb6vcc2cwf25rbdxhhm4rsxyzvtoq6dvu2bjxzo.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type, permute
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_poi_fused__to_copy_t_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/f5/cf5jpzjeahybxrklzhhb3lgj67cdslskd2rkpka3h27e5qmcjm3k.py
# Topologically Sorted Source Nodes: [q_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   q_1 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_59, %add_145], -1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x3 = xindex // 128
    x1 = ((xindex // 128) % ks0)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (384*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = (-1)*(x0)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.015625
    tmp9 = tmp7 * tmp8
    tmp10 = 10000.0
    tmp11 = libdevice.pow(tmp10, tmp9)
    tmp12 = x1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp11
    tmp15 = tl_math.cos(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp5 * tmp16
    tmp18 = tl.load(in_ptr0 + (64 + 384*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl_math.sin(tmp14)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp17 - tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr0 + (384*x3 + ((-64) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = (-1)*((-64) + x0)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 0.015625
    tmp32 = tmp30 * tmp31
    tmp33 = 10000.0
    tmp34 = libdevice.pow(tmp33, tmp32)
    tmp35 = x1
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp34
    tmp38 = tl_math.sin(tmp37)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp28 * tmp39
    tmp41 = tl.load(in_ptr0 + (64 + 384*x3 + ((-64) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl_math.cos(tmp37)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp45 = tmp40 + tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x4), tmp48, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/na/cnabpznj22s3fuj4c4shkzjhnox5d3bmdp2my5rpr3dz5c4ebjfq.py
# Topologically Sorted Source Nodes: [k_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   k_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_100, %add_241], -1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x3 = xindex // 128
    x1 = ((xindex // 128) % ks0)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 + 384*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = (-1)*(x0)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.015625
    tmp9 = tmp7 * tmp8
    tmp10 = 10000.0
    tmp11 = libdevice.pow(tmp10, tmp9)
    tmp12 = x1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp11
    tmp15 = tl_math.cos(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp5 * tmp16
    tmp18 = tl.load(in_ptr0 + (192 + 384*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl_math.sin(tmp14)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp17 - tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr0 + (128 + 384*x3 + ((-64) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = (-1)*((-64) + x0)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 0.015625
    tmp32 = tmp30 * tmp31
    tmp33 = 10000.0
    tmp34 = libdevice.pow(tmp33, tmp32)
    tmp35 = x1
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp34
    tmp38 = tl_math.sin(tmp37)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp28 * tmp39
    tmp41 = tl.load(in_ptr0 + (192 + 384*x3 + ((-64) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl_math.cos(tmp37)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp45 = tmp40 + tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x4), tmp48, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/e6/ce6kkbo5a6idxelzg2efr2t3cbv6nkuqfkxgnzp35rllwcgxiguq.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_12, permute_5
# Graph fragment:
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %permute_5 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_12, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_4 = async_compile.triton('triton_poi_fused__to_copy_t_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/zp/czpdy56lcwc2j3ab5gju2jyb7467cwcmpsa25mhk34szm2pn2em5.py
# Topologically Sorted Source Nodes: [x_1, layer_norm_1, linear_2], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   layer_norm_1 => add_303, add_304, mul_245, mul_246, rsqrt_1, sub_133, var_mean_1
#   linear_2 => convert_element_type_16
#   x_1 => add_298
# Graph fragment:
#   %add_298 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_298, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_303 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_303,), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_298, %getitem_15), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %rsqrt_1), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %primals_9), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %primals_10), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_304, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_5 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 128.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp3 - tmp13
    tmp26 = tmp25 * tmp24
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (r1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/2o/c2ovddycx4bv2wdtu2j5ttenveeruwvw4yndofnlf4b3j5eugtcg.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_15, permute_6
# Graph fragment:
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.bfloat16), kwargs = {})
#   %permute_6 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_15, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_6 = async_compile.triton('triton_poi_fused__to_copy_t_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/oy/coypfynzwsenp2zc54xvrriidetuxepwqogsvjo5bwjfcxges2fi.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_331, convert_element_type_19, convert_element_type_20, erf, mul_276, mul_277, mul_278
# Graph fragment:
#   %convert_element_type_19 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_7, torch.float32), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 0.5), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_277,), kwargs = {})
#   %add_331 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %add_331), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_278, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/tr/ctrm6yq6zq3wv3vdrplghcj5ykhhjyrcwwd4yzopicgbl4vu33kr.py
# Topologically Sorted Source Nodes: [x_1, x_2, layer_norm_2, linear_4], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_2 => add_355, add_356, mul_307, mul_308, rsqrt_2, sub_156, var_mean_2
#   linear_4 => convert_element_type_25
#   x_1 => add_298
#   x_2 => add_350
# Graph fragment:
#   %add_298 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %add_350 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_298, %view_9), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_350, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_355 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_355,), kwargs = {})
#   %sub_156 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_350, %getitem_17), kwargs = {})
#   %mul_307 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_156, %rsqrt_2), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %primals_13), kwargs = {})
#   %add_356 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %primals_14), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_356, torch.bfloat16), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp6 - tmp16
    tmp24 = 128.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 0.0078125
    tmp36 = tmp28 * tmp35
    tl.store(out_ptr2 + (r1 + 128*x0), tmp29, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp34, xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/qv/cqvxdf64mz3tysodvpvyzwdjyl5wuo4zwqfoj6ubwfquor33cxjo.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3, layer_norm_3, linear_6], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_3 => add_647, add_648, mul_544, mul_545, rsqrt_3, sub_281, var_mean_3
#   linear_6 => convert_element_type_40
#   x_1 => add_298
#   x_2 => add_350
#   x_3 => add_642
# Graph fragment:
#   %add_298 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %add_350 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_298, %view_9), kwargs = {})
#   %add_642 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_350, %view_15), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_642, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_647 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_647,), kwargs = {})
#   %sub_281 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_642, %getitem_31), kwargs = {})
#   %mul_544 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_281, %rsqrt_3), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_544, %primals_17), kwargs = {})
#   %add_648 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_545, %primals_18), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_648, torch.bfloat16), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 128.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 0.0078125
    tmp39 = tmp31 * tmp38
    tl.store(out_ptr2 + (r1 + 128*x0), tmp32, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp37, xmask)
    tl.store(out_ptr4 + (x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/x7/cx7cdjkxh4pukhfqsjq7tentduejkd6f7fpjwsajoi7pqyiuvejk.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_4 => add_699, add_700, mul_606, mul_607, rsqrt_4, sub_304, var_mean_4
#   linear_8 => convert_element_type_49
#   x_1 => add_298
#   x_2 => add_350
#   x_3 => add_642
#   x_4 => add_694
# Graph fragment:
#   %add_298 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %add_350 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_298, %view_9), kwargs = {})
#   %add_642 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_350, %view_15), kwargs = {})
#   %add_694 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_642, %view_19), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_694, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_699 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_699,), kwargs = {})
#   %sub_304 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_694, %getitem_33), kwargs = {})
#   %mul_606 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_304, %rsqrt_4), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_606, %primals_21), kwargs = {})
#   %add_700 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_607, %primals_22), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_700, torch.bfloat16), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 0.0078125
    tmp42 = tmp34 * tmp41
    tl.store(out_ptr0 + (r1 + 128*x0), tmp12, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp35, xmask)
    tl.store(out_ptr4 + (r1 + 128*x0), tmp40, xmask)
    tl.store(out_ptr5 + (x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/hx/chxn4ibhdogi4kmpvwstaiy3n5q7uksbo6tie7uhimekwmo6x4un.py
# Topologically Sorted Source Nodes: [x_5, layer_norm_5, linear_10], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_5 => add_991, add_992, mul_843, mul_844, rsqrt_5, sub_429, var_mean_5
#   linear_10 => convert_element_type_64
#   x_5 => add_986
# Graph fragment:
#   %add_986 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_694, %view_25), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_986, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_991 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_991,), kwargs = {})
#   %sub_429 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_986, %getitem_47), kwargs = {})
#   %mul_843 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_429, %rsqrt_5), kwargs = {})
#   %mul_844 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_843, %primals_25), kwargs = {})
#   %add_992 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_844, %primals_26), kwargs = {})
#   %convert_element_type_64 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_992, torch.bfloat16), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp3 - tmp13
    tmp21 = 128.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 0.0078125
    tmp33 = tmp25 * tmp32
    tl.store(out_ptr2 + (r1 + 128*x0), tmp26, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr4 + (x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/lp/clp7ydkfvzg5g76uaszeysmcicej4avkot7cppezqbj4pf6rxhtx.py
# Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8, layer_norm_8, linear_16], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_8 => add_1387, add_1388, mul_1204, mul_1205, rsqrt_8, sub_600, var_mean_8
#   linear_16 => convert_element_type_97
#   x_5 => add_986
#   x_6 => add_1038
#   x_7 => add_1330
#   x_8 => add_1382
# Graph fragment:
#   %add_986 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_694, %view_25), kwargs = {})
#   %add_1038 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_986, %view_29), kwargs = {})
#   %add_1330 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1038, %view_35), kwargs = {})
#   %add_1382 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1330, %view_39), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1382, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1387 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_64, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1387,), kwargs = {})
#   %sub_600 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1382, %getitem_65), kwargs = {})
#   %mul_1204 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_600, %rsqrt_8), kwargs = {})
#   %mul_1205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1204, %primals_37), kwargs = {})
#   %add_1388 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1205, %primals_38), kwargs = {})
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1388, torch.bfloat16), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_8, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 0.0078125
    tmp42 = tmp34 * tmp41
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp12, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp35, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp40, xmask)
    tl.store(out_ptr4 + (x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/t5/ct5xszwuavhtawvjjrcitj3y3plfnxqvgvfme7jrzs7sjesfwgiz.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, x_12, layer_norm_12, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_12 => add_2075, add_2076, mul_1802, mul_1803, rsqrt_12, sub_896, var_mean_12
#   logits => convert_element_type_145
#   x_10 => add_1726
#   x_11 => add_2018
#   x_12 => add_2070
#   x_9 => add_1674
# Graph fragment:
#   %add_1674 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1382, %view_45), kwargs = {})
#   %add_1726 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1674, %view_49), kwargs = {})
#   %add_2018 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1726, %view_55), kwargs = {})
#   %add_2070 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2018, %view_59), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2070, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2075 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_96, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2075,), kwargs = {})
#   %sub_896 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2070, %getitem_97), kwargs = {})
#   %mul_1802 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_896, %rsqrt_12), kwargs = {})
#   %mul_1803 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1802, %primals_53), kwargs = {})
#   %add_2076 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1803, %primals_54), kwargs = {})
#   %convert_element_type_145 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2076, torch.bfloat16), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_12, 128), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 0.0078125
    tmp42 = tmp34 * tmp41
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp35, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp40, xmask)
    tl.store(out_ptr3 + (x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_learning_20261020/compile_cache/1/df/cdfdpy3btawepqcplspdtw7x2cn7u6edhhzujfnixit6lridwre5.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   logits => convert_element_type_144, permute_48
# Graph fragment:
#   %convert_element_type_144 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.bfloat16), kwargs = {})
#   %permute_48 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_144, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_14 = async_compile.triton('triton_poi_fused__to_copy_t_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54 = args
    args.clear()
    s0 = primals_1
    s1 = primals_2
    assert_size_stride(primals_3, (s0, s1), (s1, 1))
    assert_size_stride(primals_4, (270, 128), (128, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (384, 128), (128, 1))
    assert_size_stride(primals_8, (128, 128), (128, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (512, 128), (128, 1))
    assert_size_stride(primals_12, (128, 512), (512, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (384, 128), (128, 1))
    assert_size_stride(primals_16, (128, 128), (128, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (512, 128), (128, 1))
    assert_size_stride(primals_20, (128, 512), (512, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (384, 128), (128, 1))
    assert_size_stride(primals_24, (128, 128), (128, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (512, 128), (128, 1))
    assert_size_stride(primals_28, (128, 512), (512, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (384, 128), (128, 1))
    assert_size_stride(primals_32, (128, 128), (128, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (512, 128), (128, 1))
    assert_size_stride(primals_36, (128, 512), (512, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (384, 128), (128, 1))
    assert_size_stride(primals_40, (128, 128), (128, 1))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (512, 128), (128, 1))
    assert_size_stride(primals_44, (128, 512), (512, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (384, 128), (128, 1))
    assert_size_stride(primals_48, (128, 128), (128, 1))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (512, 128), (128, 1))
    assert_size_stride(primals_52, (128, 512), (512, 1))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf1 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((s0, s1, 1), (s1, 1, s0*s1), torch.float32)
        buf4 = reinterpret_tensor(buf2, (s0, s1, 1), (s1, 1, 1), 0); del buf2  # reuse
        buf6 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x, layer_norm, linear], Original ATen: [aten.embedding, aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_embedding_native_layer_norm_0_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_embedding_native_layer_norm_0.run(buf4, primals_3, primals_4, primals_5, primals_6, buf0, buf1, buf6, triton_per_fused__to_copy_embedding_native_layer_norm_0_xnumel, 128, grid=grid(triton_per_fused__to_copy_embedding_native_layer_norm_0_xnumel), stream=stream0)
        del primals_6
        buf5 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_7, buf5, 49152, grid=grid(49152), stream=stream0)
        del primals_7
        buf7 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (s0*s1, 128), (128, 1), 0), buf5, out=buf7)
        buf8 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf7, buf8, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf9 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf7, buf9, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf10 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf8, buf9, reinterpret_tensor(buf7, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf11 = buf10[0]
        buf12 = buf10[1]
        buf13 = buf10[6]
        buf14 = buf10[7]
        del buf10
        buf16 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_8, buf16, 16384, grid=grid(16384), stream=stream0)
        del primals_8
        buf17 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (s0*s1, 128), (128, 1), 0), buf16, out=buf17)
        buf18 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        buf19 = empty_strided_cuda((s0, s1, 1), (s1, 1, s0*s1), torch.float32)
        buf21 = reinterpret_tensor(buf19, (s0, s1, 1), (s1, 1, 1), 0); del buf19  # reuse
        buf23 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1, layer_norm_1, linear_2], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_add_native_layer_norm_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_5.run(buf21, buf0, buf17, primals_9, primals_10, buf18, buf23, triton_per_fused__to_copy_add_native_layer_norm_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_5_xnumel), stream=stream0)
        del primals_10
        buf22 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_11, buf22, 65536, grid=grid(65536), stream=stream0)
        del primals_11
        buf24 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (s0*s1, 128), (128, 1), 0), buf22, out=buf24)
        buf25 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_12, buf25, 65536, grid=grid(65536), stream=stream0)
        del primals_12
        buf26 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf24, buf26, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf27 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (s0*s1, 512), (512, 1), 0), buf25, out=buf27)
        buf31 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf33 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf183 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, layer_norm_2, linear_4], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8.run(buf0, buf17, buf27, primals_13, primals_14, buf31, buf33, buf183, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel), stream=stream0)
        del primals_14
        buf32 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_15, buf32, 49152, grid=grid(49152), stream=stream0)
        del primals_15
        buf34 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (s0*s1, 128), (128, 1), 0), buf32, out=buf34)
        buf35 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf34, buf35, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf36 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf34, buf36, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf37 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf35, buf36, reinterpret_tensor(buf34, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf38 = buf37[0]
        buf39 = buf37[1]
        buf40 = buf37[6]
        buf41 = buf37[7]
        del buf37
        buf43 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_16, buf43, 16384, grid=grid(16384), stream=stream0)
        del primals_16
        buf44 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (s0*s1, 128), (128, 1), 0), buf43, out=buf44)
        buf48 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf50 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf182 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3, layer_norm_3, linear_6], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9.run(buf0, buf17, buf27, buf44, primals_17, primals_18, buf48, buf50, buf182, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel), stream=stream0)
        del primals_18
        buf49 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_19, buf49, 65536, grid=grid(65536), stream=stream0)
        del primals_19
        buf51 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (s0*s1, 128), (128, 1), 0), buf49, out=buf51)
        buf52 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_20, buf52, 65536, grid=grid(65536), stream=stream0)
        del primals_20
        buf53 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf51, buf53, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf54 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (s0*s1, 512), (512, 1), 0), buf52, out=buf54)
        buf55 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf59 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf61 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf181 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_4, layer_norm_4, linear_8], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10.run(buf0, buf17, buf27, buf44, buf54, primals_21, primals_22, buf55, buf59, buf61, buf181, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_10_xnumel), stream=stream0)
        del primals_22
        buf60 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_23, buf60, 49152, grid=grid(49152), stream=stream0)
        del primals_23
        buf62 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (s0*s1, 128), (128, 1), 0), buf60, out=buf62)
        buf63 = reinterpret_tensor(buf54, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [q_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf62, buf63, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf64 = reinterpret_tensor(buf44, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [k_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf62, buf64, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf65 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf63, buf64, reinterpret_tensor(buf62, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf66 = buf65[0]
        buf67 = buf65[1]
        buf68 = buf65[6]
        buf69 = buf65[7]
        del buf65
        buf71 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_24, buf71, 16384, grid=grid(16384), stream=stream0)
        del primals_24
        buf72 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (s0*s1, 128), (128, 1), 0), buf71, out=buf72)
        buf76 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf78 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf180 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, layer_norm_5, linear_10], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf55, buf72, primals_25, primals_26, buf76, buf78, buf180, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel), stream=stream0)
        del primals_26
        buf77 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_27, buf77, 65536, grid=grid(65536), stream=stream0)
        del primals_27
        buf79 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (s0*s1, 128), (128, 1), 0), buf77, out=buf79)
        buf80 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_28, buf80, 65536, grid=grid(65536), stream=stream0)
        del primals_28
        buf81 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf79, buf81, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf82 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (s0*s1, 512), (512, 1), 0), buf80, out=buf82)
        buf86 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf88 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf179 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, layer_norm_6, linear_12], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8.run(buf55, buf72, buf82, primals_29, primals_30, buf86, buf88, buf179, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel), stream=stream0)
        del primals_30
        buf87 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_31, buf87, 49152, grid=grid(49152), stream=stream0)
        del primals_31
        buf89 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (s0*s1, 128), (128, 1), 0), buf87, out=buf89)
        buf90 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf89, buf90, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf91 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf89, buf91, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf92 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf90, buf91, reinterpret_tensor(buf89, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf93 = buf92[0]
        buf94 = buf92[1]
        buf95 = buf92[6]
        buf96 = buf92[7]
        del buf92
        buf98 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_32, buf98, 16384, grid=grid(16384), stream=stream0)
        del primals_32
        buf99 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (s0*s1, 128), (128, 1), 0), buf98, out=buf99)
        buf103 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf105 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf178 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7, layer_norm_7, linear_14], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9.run(buf55, buf72, buf82, buf99, primals_33, primals_34, buf103, buf105, buf178, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel), stream=stream0)
        del primals_34
        buf104 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_35, buf104, 65536, grid=grid(65536), stream=stream0)
        del primals_35
        buf106 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (s0*s1, 128), (128, 1), 0), buf104, out=buf106)
        buf107 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_36, buf107, 65536, grid=grid(65536), stream=stream0)
        del primals_36
        buf108 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf106, buf108, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf109 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (s0*s1, 512), (512, 1), 0), buf107, out=buf109)
        buf110 = buf55; del buf55  # reuse
        buf114 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf116 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf177 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6, x_7, x_8, layer_norm_8, linear_16], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12.run(buf110, buf72, buf82, buf99, buf109, primals_37, primals_38, buf114, buf116, buf177, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12_xnumel), stream=stream0)
        del primals_38
        buf115 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_39, buf115, 49152, grid=grid(49152), stream=stream0)
        del primals_39
        buf117 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (s0*s1, 128), (128, 1), 0), buf115, out=buf117)
        buf118 = reinterpret_tensor(buf99, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf117, buf118, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf119 = reinterpret_tensor(buf82, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [k_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf117, buf119, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf120 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf118, buf119, reinterpret_tensor(buf117, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf121 = buf120[0]
        buf122 = buf120[1]
        buf123 = buf120[6]
        buf124 = buf120[7]
        del buf120
        buf126 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_40, buf126, 16384, grid=grid(16384), stream=stream0)
        del primals_40
        buf127 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (s0*s1, 128), (128, 1), 0), buf126, out=buf127)
        buf131 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf133 = reinterpret_tensor(buf109, (s0, s1, 128), (128*s1, 128, 1), 0); del buf109  # reuse
        buf176 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, layer_norm_9, linear_18], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf110, buf127, primals_41, primals_42, buf131, buf133, buf176, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_xnumel), stream=stream0)
        del primals_42
        buf132 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_43, buf132, 65536, grid=grid(65536), stream=stream0)
        del primals_43
        buf134 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (s0*s1, 128), (128, 1), 0), buf132, out=buf134)
        buf135 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_44, buf135, 65536, grid=grid(65536), stream=stream0)
        del primals_44
        buf136 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf134, buf136, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf137 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (s0*s1, 512), (512, 1), 0), buf135, out=buf137)
        buf141 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf143 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf175 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_10, layer_norm_10, linear_20], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8.run(buf110, buf127, buf137, primals_45, primals_46, buf141, buf143, buf175, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8_xnumel), stream=stream0)
        del primals_46
        buf142 = empty_strided_cuda((128, 384), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_47, buf142, 49152, grid=grid(49152), stream=stream0)
        del primals_47
        buf144 = empty_strided_cuda((s0*s1, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (s0*s1, 128), (128, 1), 0), buf142, out=buf144)
        buf145 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf144, buf145, s1, triton_poi_fused_cat_2_xnumel, grid=grid(triton_poi_fused_cat_2_xnumel), stream=stream0)
        buf146 = empty_strided_cuda((s0, 1, s1, 128), (128*s1, 128*s1, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf144, buf146, s1, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf147 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf145, buf146, reinterpret_tensor(buf144, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), 0.0, True, scale=0.08838834764831843)
        buf148 = buf147[0]
        buf149 = buf147[1]
        buf150 = buf147[6]
        buf151 = buf147[7]
        del buf147
        buf153 = empty_strided_cuda((128, 128), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_4.run(primals_48, buf153, 16384, grid=grid(16384), stream=stream0)
        del primals_48
        buf154 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (s0*s1, 128), (128, 1), 0), buf153, out=buf154)
        buf158 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf160 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf174 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_10, x_11, layer_norm_11, linear_22], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9.run(buf110, buf127, buf137, buf154, primals_49, primals_50, buf158, buf160, buf174, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_9_xnumel), stream=stream0)
        del primals_50
        buf159 = empty_strided_cuda((128, 512), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_51, buf159, 65536, grid=grid(65536), stream=stream0)
        del primals_51
        buf161 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (s0*s1, 128), (128, 1), 0), buf159, out=buf161)
        buf162 = empty_strided_cuda((512, 128), (1, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_52, buf162, 65536, grid=grid(65536), stream=stream0)
        del primals_52
        buf163 = empty_strided_cuda((s0, s1, 512), (512*s1, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf161, buf163, triton_poi_fused_gelu_7_xnumel, grid=grid(triton_poi_fused_gelu_7_xnumel), stream=stream0)
        buf164 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (s0*s1, 512), (512, 1), 0), buf162, out=buf164)
        buf165 = buf110; del buf110  # reuse
        buf169 = buf165; del buf165  # reuse
        buf171 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.bfloat16)
        buf173 = empty_strided_cuda((s0, s1, 1), (s1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, x_10, x_11, x_12, layer_norm_12, logits], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13.run(buf169, buf127, buf137, buf154, buf164, primals_53, primals_54, buf171, buf173, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13_xnumel), stream=stream0)
        del buf127
        del buf137
        del buf154
        del buf164
        del primals_54
        buf170 = empty_strided_cuda((128, 270), (1, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_14.run(primals_4, buf170, 34560, grid=grid(34560), stream=stream0)
        del primals_4
        buf172 = empty_strided_cuda((s0*s1, 270), (270, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (s0*s1, 128), (128, 1), 0), buf170, out=buf172)
    return (reinterpret_tensor(buf172, (s0, s1, 270), (270*s1, 270, 1), 0), primals_3, primals_5, primals_9, primals_13, primals_17, primals_21, primals_25, primals_29, primals_33, primals_37, primals_41, primals_45, primals_49, primals_53, buf0, buf1, buf4, reinterpret_tensor(buf6, (s0*s1, 128), (128, 1), 0), buf8, buf9, reinterpret_tensor(buf7, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf11, buf12, buf13, buf14, buf17, buf18, buf21, reinterpret_tensor(buf23, (s0*s1, 128), (128, 1), 0), buf24, reinterpret_tensor(buf26, (s0*s1, 512), (512, 1), 0), buf31, reinterpret_tensor(buf33, (s0*s1, 128), (128, 1), 0), buf35, buf36, reinterpret_tensor(buf34, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf38, buf39, buf40, buf41, buf48, reinterpret_tensor(buf50, (s0*s1, 128), (128, 1), 0), buf51, reinterpret_tensor(buf53, (s0*s1, 512), (512, 1), 0), buf59, reinterpret_tensor(buf61, (s0*s1, 128), (128, 1), 0), buf63, buf64, reinterpret_tensor(buf62, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf66, buf67, buf68, buf69, buf76, reinterpret_tensor(buf78, (s0*s1, 128), (128, 1), 0), buf79, reinterpret_tensor(buf81, (s0*s1, 512), (512, 1), 0), buf86, reinterpret_tensor(buf88, (s0*s1, 128), (128, 1), 0), buf90, buf91, reinterpret_tensor(buf89, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf93, buf94, buf95, buf96, buf103, reinterpret_tensor(buf105, (s0*s1, 128), (128, 1), 0), buf106, reinterpret_tensor(buf108, (s0*s1, 512), (512, 1), 0), buf114, reinterpret_tensor(buf116, (s0*s1, 128), (128, 1), 0), buf118, buf119, reinterpret_tensor(buf117, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf121, buf122, buf123, buf124, buf131, reinterpret_tensor(buf133, (s0*s1, 128), (128, 1), 0), buf134, reinterpret_tensor(buf136, (s0*s1, 512), (512, 1), 0), buf141, reinterpret_tensor(buf143, (s0*s1, 128), (128, 1), 0), buf145, buf146, reinterpret_tensor(buf144, (s0, 1, s1, 128), (384*s1, 128, 384, 1), 256), buf148, buf149, buf150, buf151, buf158, reinterpret_tensor(buf160, (s0*s1, 128), (128, 1), 0), buf161, reinterpret_tensor(buf163, (s0*s1, 512), (512, 1), 0), buf169, reinterpret_tensor(buf171, (s0*s1, 128), (128, 1), 0), reinterpret_tensor(buf170, (270, 128), (128, 1), 0), buf173, reinterpret_tensor(buf162, (128, 512), (512, 1), 0), reinterpret_tensor(buf159, (512, 128), (128, 1), 0), buf174, reinterpret_tensor(buf153, (128, 128), (128, 1), 0), reinterpret_tensor(buf142, (384, 128), (128, 1), 0), buf175, reinterpret_tensor(buf135, (128, 512), (512, 1), 0), reinterpret_tensor(buf132, (512, 128), (128, 1), 0), buf176, reinterpret_tensor(buf126, (128, 128), (128, 1), 0), reinterpret_tensor(buf115, (384, 128), (128, 1), 0), buf177, reinterpret_tensor(buf107, (128, 512), (512, 1), 0), reinterpret_tensor(buf104, (512, 128), (128, 1), 0), buf178, reinterpret_tensor(buf98, (128, 128), (128, 1), 0), reinterpret_tensor(buf87, (384, 128), (128, 1), 0), buf179, reinterpret_tensor(buf80, (128, 512), (512, 1), 0), reinterpret_tensor(buf77, (512, 128), (128, 1), 0), buf180, reinterpret_tensor(buf71, (128, 128), (128, 1), 0), reinterpret_tensor(buf60, (384, 128), (128, 1), 0), buf181, reinterpret_tensor(buf52, (128, 512), (512, 1), 0), reinterpret_tensor(buf49, (512, 128), (128, 1), 0), buf182, reinterpret_tensor(buf43, (128, 128), (128, 1), 0), reinterpret_tensor(buf32, (384, 128), (128, 1), 0), buf183, reinterpret_tensor(buf25, (128, 512), (512, 1), 0), reinterpret_tensor(buf22, (512, 128), (128, 1), 0), reinterpret_tensor(buf16, (128, 128), (128, 1), 0), reinterpret_tensor(buf5, (384, 128), (128, 1), 0), s0, s1, s0*s1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 64
    primals_2 = 10
    primals_3 = rand_strided((64, 10), (10, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((270, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
