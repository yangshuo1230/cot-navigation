# AOT ID: ['0_backward']
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


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/b6/cb6phkaieccroduh7carhgwrxls7iniik752rvj5hcm3usykpkyh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_152 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_63, torch.float32), kwargs = {})
#   %mul_1834 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %primals_53), kwargs = {})
#   %mul_1835 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1834, 128), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1834, [2], True), kwargs = {})
#   %mul_1836 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1834, %mul_1802), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1836, [2], True), kwargs = {})
#   %mul_1837 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1802, %sum_2), kwargs = {})
#   %sub_910 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1835, %sum_1), kwargs = {})
#   %sub_911 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_910, %mul_1837), kwargs = {})
#   %mul_1838 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sub_911), kwargs = {})
#   %convert_element_type_154 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1838, torch.bfloat16), kwargs = {})
#   %view_64 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_154, [%mul_20, 128]), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_backward_view_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_backward_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 - tmp7
    tmp18 = tmp8 * tmp13
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp20, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/fb/cfbawab6wmkme6lwbjr5i4voji5ytrmu6532nk5hw7usyjz6rotk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_152 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_63, torch.float32), kwargs = {})
#   %mul_1839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %mul_1802), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1839, [0, 1]), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_152, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_1 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 128
    x0 = (xindex % 128)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*((37 + ks0*ks1) // 38)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp13 = tl.where(tmp2, tmp4, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/vs/cvswrwbcwqypeal47yz5ic4c7w4qxjk6nkpv7djtjtv2xry3fldd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_152 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_63, torch.float32), kwargs = {})
#   %mul_1839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %mul_1802), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1839, [0, 1]), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_backward_2 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_backward_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 38
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r1), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/af/cafbaxoxiac6sc47725gyqkc23ye7voasvnxtsyfdcssh5gfyfr2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_159 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_27, torch.float32), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/gi/cgim22te6kgeczcbjtbiyc7izivt3xulikhvwdewvlqmocdjpods.py
# Topologically Sorted Source Nodes: [gelu_5], Original ATen: [aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   gelu_5 => add_2051, convert_element_type_139, erf_5, mul_1772
# Graph fragment:
#   %convert_element_type_160 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_65, torch.float32), kwargs = {})
#   %convert_element_type_139 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_57, torch.float32), kwargs = {})
#   %mul_1772 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_139, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1772,), kwargs = {})
#   %add_2051 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_1841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2051, 0.5), kwargs = {})
#   %mul_1842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_139, %convert_element_type_139), kwargs = {})
#   %mul_1843 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1842, -0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1843,), kwargs = {})
#   %mul_1844 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_1845 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_139, %mul_1844), kwargs = {})
#   %add_2104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1841, %mul_1845), kwargs = {})
#   %mul_1846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_160, %add_2104), kwargs = {})
#   %convert_element_type_162 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1846, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_gelu_backward_4 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/m3/cm3mmligslfiojlvjmhbsex6pyahodecwuatmgzvbvulykpoboqh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_167 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_67, torch.float32), kwargs = {})
#   %mul_1848 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_167, %primals_49), kwargs = {})
#   %mul_1849 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1848, 128), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1848, [2], True), kwargs = {})
#   %mul_1850 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1848, %mul_1740), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1850, [2], True), kwargs = {})
#   %mul_1851 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1740, %sum_6), kwargs = {})
#   %sub_913 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1849, %sum_5), kwargs = {})
#   %sub_914 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_913, %mul_1851), kwargs = {})
#   %mul_1852 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %sub_914), kwargs = {})
#   %add_2105 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1838, %mul_1852), kwargs = {})
#   %convert_element_type_169 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2105, torch.bfloat16), kwargs = {})
#   %view_68 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_169, [%mul_20, 128]), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_backward_view_5 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_backward_view_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_backward_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_backward_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0)
    tmp14 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp16 = 128.0
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 - tmp7
    tmp19 = tmp8 * tmp13
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp14 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp22, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/54/c54fo7bj7pt4le4l2fgdbquziuqq5h4ifswdqzklh75att3m2oi7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_174 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_31, torch.float32), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/mm/cmmvs6dfuc4dmyri357fzs3ajj75x7uq7h5phznj2i2n5z6u4sf5.py
# Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.neg, aten.add, aten.slice_backward]
# Source node to ATen node mapping:
#   angles => mul_68
#   cos => cos
#   cos_1 => convert_element_type_6
#   sin => sin
#   sin_1 => convert_element_type_7
# Graph fragment:
#   %mul_68 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, %unsqueeze_5), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_68,), kwargs = {})
#   %convert_element_type_6 : [num_users=24] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cos, torch.bfloat16), kwargs = {})
#   %mul_1854 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_62, %convert_element_type_6), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_68,), kwargs = {})
#   %convert_element_type_7 : [num_users=24] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sin, torch.bfloat16), kwargs = {})
#   %mul_1855 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_62, %convert_element_type_7), kwargs = {})
#   %neg_12 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_61,), kwargs = {})
#   %mul_1856 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_12, %convert_element_type_7), kwargs = {})
#   %add_2106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1854, %mul_1856), kwargs = {})
#   %mul_1857 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_61, %convert_element_type_6), kwargs = {})
#   %add_2107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1855, %mul_1857), kwargs = {})
#   %full_default : [num_users=24] = call_function[target=torch.ops.aten.full.default](args = ([%primals_1, 1, %primals_2, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %add_2106, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %add_2107, 3, 0, 64), kwargs = {})
#   %add_2108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7 = async_compile.triton('triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x3 = xindex
    x1 = ((xindex // 128) % ks0)
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = 64 + ((-1)*x0)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.015625
    tmp7 = tmp5 * tmp6
    tmp8 = 10000.0
    tmp9 = libdevice.pow(tmp8, tmp7)
    tmp10 = x1
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp9
    tmp13 = tl_math.cos(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp3 * tmp14
    tmp16 = tl.load(in_ptr0 + ((-64) + x3), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = -tmp16
    tmp18 = tl_math.sin(tmp12)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp15 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = 0.0
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tmp0 < tmp1
    tmp27 = tl.load(in_ptr0 + (64 + x3), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = (-1)*x0
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 0.015625
    tmp31 = tmp29 * tmp30
    tmp32 = 10000.0
    tmp33 = libdevice.pow(tmp32, tmp31)
    tmp34 = x1
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp33
    tmp37 = tl_math.sin(tmp36)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp27 * tmp38
    tmp40 = tl.load(in_ptr0 + (x3), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl_math.cos(tmp36)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 * tmp42
    tmp44 = tmp39 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp26, tmp44, tmp45)
    tmp47 = tl.where(tmp26, tmp46, tmp24)
    tmp48 = tmp25 + tmp47
    tl.store(out_ptr0 + (x3), tmp48, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/xg/cxg63tzkql6v7s4jvls3kk5b3zlomectknml5czt46crjjzlhbnw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_68, %permute_67, %permute_66], 2), kwargs = {})
triton_poi_fused_stack_8 = async_compile.triton('triton_poi_fused_stack_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 128) % 3)
    x0 = (xindex % 128)
    x2 = xindex // 384
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 128*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 128*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 128*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/4o/c4o6azm5xbrbqggjubx4y4ydcopcurbgsxq7exm5x4tn35cnq5p3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_180 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_33, torch.float32), kwargs = {})
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/3e/c3eeq7kkimrngj4gj7bbfakk2yh2amakbrf3pul6qwphoaunjybw.py
# Topologically Sorted Source Nodes: [x_1, layer_norm_1], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.view]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_245, sub_133
#   x_1 => add_298
# Graph fragment:
#   %convert_element_type_302 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_122, torch.float32), kwargs = {})
#   %mul_1993 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_302, %primals_9), kwargs = {})
#   %mul_1994 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1993, 128), kwargs = {})
#   %sum_45 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1993, [2], True), kwargs = {})
#   %add_298 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_298, %getitem_15), kwargs = {})
#   %mul_245 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %rsqrt_1), kwargs = {})
#   %mul_1995 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1993, %mul_245), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1995, [2], True), kwargs = {})
#   %mul_1996 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %sum_46), kwargs = {})
#   %sub_943 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1994, %sum_45), kwargs = {})
#   %sub_944 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_943, %mul_1996), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_1997 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_23, %sub_944), kwargs = {})
#   %add_2155 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2152, %mul_1997), kwargs = {})
#   %convert_element_type_304 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2155, torch.bfloat16), kwargs = {})
#   %view_123 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_304, [%mul_20, 128]), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp3 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp22 = 0.0078125
    tmp23 = tmp14 * tmp22
    tmp24 = 128.0
    tmp25 = tmp3 * tmp24
    tmp26 = tmp25 - tmp7
    tmp27 = tmp15 * tmp20
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp21 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp30, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/fr/cfrp5ivnhii6wei47yzondbduhtejftcmuw3mx62qru6mmclfsye.py
# Topologically Sorted Source Nodes: [x_1, layer_norm_1, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => mul_8, sub_8
#   layer_norm_1 => mul_245, sub_133
#   x_1 => add_298
# Graph fragment:
#   %convert_element_type_302 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_122, torch.float32), kwargs = {})
#   %add_298 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_5), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_298, %getitem_15), kwargs = {})
#   %mul_245 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %rsqrt_1), kwargs = {})
#   %mul_1998 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_302, %mul_245), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1998, [0, 1]), kwargs = {})
#   %sum_48 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_302, [0, 1]), kwargs = {})
#   %convert_element_type_314 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_129, torch.float32), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %getitem_1), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt), kwargs = {})
#   %mul_2013 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_314, %mul_8), kwargs = {})
#   %sum_51 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2013, [0, 1]), kwargs = {})
#   %sum_52 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_314, [0, 1]), kwargs = {})
triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 128
    x0 = (xindex % 128)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp39 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*((37 + ks0*ks1) // 38)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp9 = tl.load(in_ptr3 + (((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tl.load(in_ptr4 + (((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr5 + (x0 + 128*(((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.load(in_ptr6 + (((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp5 - tmp21
        tmp23 = tl.load(in_ptr7 + (((r2 + x1*((37 + ks0*ks1) // 38)) % (ks0*ks1))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 * tmp23
        tmp25 = tmp20 * tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
        tmp31 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp32 = tl.where(tmp2, tmp4, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
        tmp36 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp37 = tl.where(tmp2, tmp20, tmp36)
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(rmask & xmask, tmp40, _tmp39)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
    tl.store(out_ptr2 + (x3), tmp34, xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/jk/cjkhx7da6gyn4unl4b5z5bf2zdoevft3ukgimtnlixj7lfkqxo7e.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_25 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([270, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_12 = async_compile.triton('triton_poi_fused_embedding_dense_backward_12', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/oy/coyjfmzfb756afo3vhxh62n7zba4cha5dhhnseuvdtvsjsevben7.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   layer_norm => mul_8, sub_8
# Graph fragment:
#   %convert_element_type_314 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_129, torch.float32), kwargs = {})
#   %mul_2008 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_314, %primals_5), kwargs = {})
#   %mul_2009 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2008, 128), kwargs = {})
#   %sum_49 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2008, [2], True), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %getitem_1), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt), kwargs = {})
#   %mul_2010 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2008, %mul_8), kwargs = {})
#   %sum_50 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2010, [2], True), kwargs = {})
#   %mul_2011 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %sum_50), kwargs = {})
#   %sub_946 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2009, %sum_49), kwargs = {})
#   %sub_947 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_946, %mul_2011), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_2012 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_24, %sub_947), kwargs = {})
#   %add_2162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2155, %mul_2012), kwargs = {})
#   %full_default_24 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_61, %full_default_24, %add_2162), kwargs = {})
#   %full_default_25 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([270, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_25, [%primals_3], %where, True), kwargs = {})
triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (r1 + 128*x0), xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tl.full([1, 1], 0, tl.int64)
    tmp20 = tmp18 == tmp19
    tmp22 = 0.0078125
    tmp23 = tmp11 * tmp22
    tmp24 = 128.0
    tmp25 = tmp3 * tmp24
    tmp26 = tmp25 - tmp7
    tmp27 = tmp12 * tmp17
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp21 + tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp20, tmp31, tmp30)
    tmp33 = tl.full([XBLOCK, RBLOCK], 270, tl.int32)
    tmp34 = tmp18 + tmp33
    tmp35 = tmp18 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp18)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 270)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 270")
    tl.atomic_add(out_ptr2 + (tl.broadcast_to(r1 + 128*tmp36, [XBLOCK, RBLOCK])), tmp32, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /mnt/ningxuefei/yangshuo/cot-navigation-unified/research/direct_multihop_20261021/compile_cache/5/jg/cjgugycdnpkhoqlpwgbqjw3aamv6kpktgwqn6zprhfd7ibluv6wl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_153 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_25, torch.float32), kwargs = {})
#   %add_2163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_153, %index_put), kwargs = {})
triton_poi_fused__to_copy_add_14 = async_compile.triton('triton_poi_fused__to_copy_add_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C60067E633ACDD066E9D817B2C3DDDB23A18874CE8A1BD1EB2E51156FF8E197D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, mul_20, primals_3, primals_5, primals_9, primals_13, primals_17, primals_21, primals_25, primals_29, primals_33, primals_37, primals_41, primals_45, primals_49, primals_53, embedding, getitem_1, rsqrt, view, cat, cat_1, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mm_1, getitem_15, rsqrt_1, view_6, mm_2, view_8, mul_307, view_10, cat_2, cat_3, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_544, view_16, mm_6, view_18, mul_606, view_20, cat_4, cat_5, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_843, view_26, mm_10, view_28, mul_905, view_30, cat_6, cat_7, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_1142, view_36, mm_14, view_38, mul_1204, view_40, cat_8, cat_9, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_1441, view_46, mm_18, view_48, mul_1503, view_50, cat_10, cat_11, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_1740, view_56, mm_22, view_58, mul_1802, view_60, permute_51, div_12, permute_55, permute_59, div_13, permute_63, permute_71, div_14, permute_75, permute_79, div_15, permute_83, permute_91, div_16, permute_95, permute_99, div_17, permute_103, permute_111, div_18, permute_115, permute_119, div_19, permute_123, permute_131, div_20, permute_135, permute_139, div_21, permute_143, permute_151, div_22, permute_155, permute_159, permute_163, permute_171, tangents_1 = args
    args.clear()
    s0 = primals_1
    s1 = primals_2
    assert_size_stride(primals_3, (s0, s1), (s1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(embedding, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(getitem_1, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(rsqrt, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(view, (s0*s1, 128), (128, 1))
    assert_size_stride(cat, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_1, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_3, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_5, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_6, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(getitem_12, (), ())
    assert_size_stride(mm_1, (s0*s1, 128), (128, 1))
    assert_size_stride(getitem_15, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(rsqrt_1, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(view_6, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_2, (s0*s1, 512), (512, 1))
    assert_size_stride(view_8, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_307, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_10, (s0*s1, 128), (128, 1))
    assert_size_stride(cat_2, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_3, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_11, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_21, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_22, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(getitem_28, (), ())
    assert_size_stride(mul_544, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_16, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_6, (s0*s1, 512), (512, 1))
    assert_size_stride(view_18, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_606, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_20, (s0*s1, 128), (128, 1))
    assert_size_stride(cat_4, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_5, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_19, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_37, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_38, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_43, (), ())
    assert_size_stride(getitem_44, (), ())
    assert_size_stride(mul_843, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_26, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_10, (s0*s1, 512), (512, 1))
    assert_size_stride(view_28, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_905, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_30, (s0*s1, 128), (128, 1))
    assert_size_stride(cat_6, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_7, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_27, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_53, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_54, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(mul_1142, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_36, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_14, (s0*s1, 512), (512, 1))
    assert_size_stride(view_38, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_1204, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_40, (s0*s1, 128), (128, 1))
    assert_size_stride(cat_8, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_9, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_35, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_69, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_70, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_75, (), ())
    assert_size_stride(getitem_76, (), ())
    assert_size_stride(mul_1441, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_46, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_18, (s0*s1, 512), (512, 1))
    assert_size_stride(view_48, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_1503, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_50, (s0*s1, 128), (128, 1))
    assert_size_stride(cat_10, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(cat_11, (s0, 1, s1, 128), (128*s1, 128*s1, 128, 1))
    assert_size_stride(permute_43, (s0, 1, s1, 128), (384*s1, 128, 384, 1))
    assert_size_stride(getitem_85, (s0, 1, s1, 128), (128*s1, 128, 128, 1))
    assert_size_stride(getitem_86, (s0, 1, s1), (s1, s1, 1))
    assert_size_stride(getitem_91, (), ())
    assert_size_stride(getitem_92, (), ())
    assert_size_stride(mul_1740, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_56, (s0*s1, 128), (128, 1))
    assert_size_stride(mm_22, (s0*s1, 512), (512, 1))
    assert_size_stride(view_58, (s0*s1, 512), (512, 1))
    assert_size_stride(mul_1802, (s0, s1, 128), (128*s1, 128, 1))
    assert_size_stride(view_60, (s0*s1, 128), (128, 1))
    assert_size_stride(permute_51, (270, 128), (128, 1))
    assert_size_stride(div_12, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_55, (128, 512), (512, 1))
    assert_size_stride(permute_59, (512, 128), (128, 1))
    assert_size_stride(div_13, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_63, (128, 128), (128, 1))
    assert_size_stride(permute_71, (384, 128), (128, 1))
    assert_size_stride(div_14, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_75, (128, 512), (512, 1))
    assert_size_stride(permute_79, (512, 128), (128, 1))
    assert_size_stride(div_15, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_83, (128, 128), (128, 1))
    assert_size_stride(permute_91, (384, 128), (128, 1))
    assert_size_stride(div_16, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_95, (128, 512), (512, 1))
    assert_size_stride(permute_99, (512, 128), (128, 1))
    assert_size_stride(div_17, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_103, (128, 128), (128, 1))
    assert_size_stride(permute_111, (384, 128), (128, 1))
    assert_size_stride(div_18, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_115, (128, 512), (512, 1))
    assert_size_stride(permute_119, (512, 128), (128, 1))
    assert_size_stride(div_19, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_123, (128, 128), (128, 1))
    assert_size_stride(permute_131, (384, 128), (128, 1))
    assert_size_stride(div_20, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_135, (128, 512), (512, 1))
    assert_size_stride(permute_139, (512, 128), (128, 1))
    assert_size_stride(div_21, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_143, (128, 128), (128, 1))
    assert_size_stride(permute_151, (384, 128), (128, 1))
    assert_size_stride(div_22, (s0, s1, 1), (s1, 1, 1))
    assert_size_stride(permute_155, (128, 512), (512, 1))
    assert_size_stride(permute_159, (512, 128), (128, 1))
    assert_size_stride(permute_163, (128, 128), (128, 1))
    assert_size_stride(permute_171, (384, 128), (128, 1))
    assert_size_stride(tangents_1, (s0, s1, 270), (270*s1, 270, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((270, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (270, s0*s1), (1, 270), 0), view_60, out=buf0)
        del view_60
        buf1 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (s0*s1, 270), (270, 1), 0), permute_51, out=buf1)
        del permute_51
        del tangents_1
        buf4 = empty_strided_cuda((s0, s1, 128), (128*s1, 128, 1), torch.float32)
        buf9 = empty_strided_cuda((s0*s1, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.view]
        triton_per_fused__to_copy_native_layer_norm_backward_view_0_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_view_0.run(buf1, primals_53, mul_1802, div_12, buf4, buf9, triton_per_fused__to_copy_native_layer_norm_backward_view_0_xnumel, 128, grid=grid(triton_per_fused__to_copy_native_layer_norm_backward_view_0_xnumel), stream=stream0)
        del div_12
        del primals_53
        buf5 = empty_strided_cuda((128, 38), (1, 128), torch.float32)
        buf7 = empty_strided_cuda((128, 38), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf1, mul_1802, buf5, buf7, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1802
        buf6 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf5, buf6, 128, 38, grid=grid(128), stream=stream0)
        buf8 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf7, buf8, 128, 38, grid=grid(128), stream=stream0)
        buf10 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (128, s0*s1), (1, 128), 0), view_58, out=buf10)
        del view_58
        buf11 = empty_strided_cuda((s0*s1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, permute_55, out=buf11)
        del permute_55
        buf12 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf10, buf12, 65536, grid=grid(65536), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (s0, s1, 512), (512*s1, 512, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [gelu_5], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf13, mm_22, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_22
        buf14 = reinterpret_tensor(buf10, (512, 128), (128, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, s0*s1), (1, 512), 0), view_56, out=buf14)
        del view_56
        buf15 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (s0*s1, 512), (512, 1), 0), permute_59, out=buf15)
        del permute_59
        buf16 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf14, buf16, 65536, grid=grid(65536), stream=stream0)
        buf23 = buf4; del buf4  # reuse
        buf24 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf23, buf15, primals_49, mul_1740, div_13, buf24, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_13
        del primals_49
        buf19 = buf7; del buf7  # reuse
        buf21 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf15, mul_1740, buf19, buf21, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1740
        buf20 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf19, buf20, 128, 38, grid=grid(128), stream=stream0)
        buf22 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf21, buf22, 128, 38, grid=grid(128), stream=stream0)
        buf25 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_85, (s0*s1, 128), (128, 1), 0), out=buf25)
        buf26 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, permute_63, out=buf26)
        del buf24
        del permute_63
        buf27 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf25, buf27, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf28 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf26, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat_10, cat_11, permute_43, getitem_85, getitem_86, None, None, s1, s1, 0.0, True, getitem_91, getitem_92, scale=0.08838834764831843)
        del cat_10
        del cat_11
        del getitem_85
        del getitem_86
        del getitem_91
        del getitem_92
        del permute_43
        buf29 = buf28[0]
        buf30 = buf28[1]
        buf31 = buf28[2]
        del buf28
        buf32 = reinterpret_tensor(buf26, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.neg, aten.add, aten.slice_backward]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf30, buf32, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf33 = reinterpret_tensor(buf30, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf29, buf33, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf29
        buf34 = empty_strided_cuda((s0, s1, 3, 128), (384*s1, 384, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf33, buf32, buf31, buf34, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        del buf31
        buf35 = empty_strided_cuda((384, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (384, s0*s1), (1, 384), 0), view_50, out=buf35)
        del view_50
        buf36 = reinterpret_tensor(buf33, (s0*s1, 128), (128, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (s0*s1, 384), (384, 1), 0), permute_71, out=buf36)
        del permute_71
        buf37 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf35, buf37, 49152, grid=grid(49152), stream=stream0)
        buf44 = buf23; del buf23  # reuse
        buf45 = reinterpret_tensor(buf32, (s0*s1, 128), (128, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf44, buf36, primals_45, mul_1503, div_14, buf45, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_14
        del primals_45
        buf40 = buf21; del buf21  # reuse
        buf42 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf36, mul_1503, buf40, buf42, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1503
        buf41 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf40, buf41, 128, 38, grid=grid(128), stream=stream0)
        buf43 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf42, buf43, 128, 38, grid=grid(128), stream=stream0)
        buf46 = reinterpret_tensor(buf14, (128, 512), (512, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (128, s0*s1), (1, 128), 0), view_48, out=buf46)
        del view_48
        buf47 = reinterpret_tensor(buf13, (s0*s1, 512), (512, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, permute_75, out=buf47)
        del permute_75
        buf48 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf46, buf48, 65536, grid=grid(65536), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (s0, s1, 512), (512*s1, 512, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [gelu_4], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf49, mm_18, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_18
        buf50 = reinterpret_tensor(buf46, (512, 128), (128, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, s0*s1), (1, 512), 0), view_46, out=buf50)
        del view_46
        buf51 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (s0*s1, 512), (512, 1), 0), permute_79, out=buf51)
        del permute_79
        buf52 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf50, buf52, 65536, grid=grid(65536), stream=stream0)
        buf59 = buf44; del buf44  # reuse
        buf60 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf59, buf51, primals_41, mul_1441, div_15, buf60, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_15
        del primals_41
        buf55 = buf42; del buf42  # reuse
        buf57 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf51, mul_1441, buf55, buf57, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1441
        buf56 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf55, buf56, 128, 38, grid=grid(128), stream=stream0)
        buf58 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf57, buf58, 128, 38, grid=grid(128), stream=stream0)
        buf61 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_69, (s0*s1, 128), (128, 1), 0), out=buf61)
        buf62 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf60, permute_83, out=buf62)
        del buf60
        del permute_83
        buf63 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf61, buf63, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf64 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf62, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat_8, cat_9, permute_35, getitem_69, getitem_70, None, None, s1, s1, 0.0, True, getitem_75, getitem_76, scale=0.08838834764831843)
        del cat_8
        del cat_9
        del getitem_69
        del getitem_70
        del getitem_75
        del getitem_76
        del permute_35
        buf65 = buf64[0]
        buf66 = buf64[1]
        buf67 = buf64[2]
        del buf64
        buf68 = reinterpret_tensor(buf62, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf66, buf68, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf69 = reinterpret_tensor(buf66, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf65, buf69, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf65
        buf70 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf69, buf68, buf67, buf70, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        del buf67
        buf71 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (384, s0*s1), (1, 384), 0), view_40, out=buf71)
        del view_40
        buf72 = reinterpret_tensor(buf69, (s0*s1, 128), (128, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (s0*s1, 384), (384, 1), 0), permute_91, out=buf72)
        del permute_91
        buf73 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf71, buf73, 49152, grid=grid(49152), stream=stream0)
        buf80 = buf59; del buf59  # reuse
        buf81 = reinterpret_tensor(buf68, (s0*s1, 128), (128, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf80, buf72, primals_37, mul_1204, div_16, buf81, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_16
        del primals_37
        buf76 = buf57; del buf57  # reuse
        buf78 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf72, mul_1204, buf76, buf78, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1204
        buf77 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf76, buf77, 128, 38, grid=grid(128), stream=stream0)
        buf79 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf78, buf79, 128, 38, grid=grid(128), stream=stream0)
        buf82 = reinterpret_tensor(buf50, (128, 512), (512, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (128, s0*s1), (1, 128), 0), view_38, out=buf82)
        del view_38
        buf83 = reinterpret_tensor(buf49, (s0*s1, 512), (512, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf81, permute_95, out=buf83)
        del permute_95
        buf84 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf82, buf84, 65536, grid=grid(65536), stream=stream0)
        buf85 = reinterpret_tensor(buf83, (s0, s1, 512), (512*s1, 512, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [gelu_3], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf85, mm_14, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_14
        buf86 = reinterpret_tensor(buf82, (512, 128), (128, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, s0*s1), (1, 512), 0), view_36, out=buf86)
        del view_36
        buf87 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (s0*s1, 512), (512, 1), 0), permute_99, out=buf87)
        del permute_99
        buf88 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf86, buf88, 65536, grid=grid(65536), stream=stream0)
        buf95 = buf80; del buf80  # reuse
        buf96 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf95, buf87, primals_33, mul_1142, div_17, buf96, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_17
        del primals_33
        buf91 = buf78; del buf78  # reuse
        buf93 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf87, mul_1142, buf91, buf93, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_1142
        buf92 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf91, buf92, 128, 38, grid=grid(128), stream=stream0)
        buf94 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf93, buf94, 128, 38, grid=grid(128), stream=stream0)
        buf97 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_53, (s0*s1, 128), (128, 1), 0), out=buf97)
        buf98 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf96, permute_103, out=buf98)
        del buf96
        del permute_103
        buf99 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf97, buf99, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf100 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf98, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat_6, cat_7, permute_27, getitem_53, getitem_54, None, None, s1, s1, 0.0, True, getitem_59, getitem_60, scale=0.08838834764831843)
        del cat_6
        del cat_7
        del getitem_53
        del getitem_54
        del getitem_59
        del getitem_60
        del permute_27
        buf101 = buf100[0]
        buf102 = buf100[1]
        buf103 = buf100[2]
        del buf100
        buf104 = reinterpret_tensor(buf98, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf102, buf104, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf105 = reinterpret_tensor(buf102, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf101, buf105, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf101
        buf106 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf105, buf104, buf103, buf106, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        del buf103
        buf107 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (384, s0*s1), (1, 384), 0), view_30, out=buf107)
        del view_30
        buf108 = reinterpret_tensor(buf105, (s0*s1, 128), (128, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (s0*s1, 384), (384, 1), 0), permute_111, out=buf108)
        del permute_111
        buf109 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf107, buf109, 49152, grid=grid(49152), stream=stream0)
        buf116 = buf95; del buf95  # reuse
        buf117 = reinterpret_tensor(buf104, (s0*s1, 128), (128, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf116, buf108, primals_29, mul_905, div_18, buf117, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_18
        del primals_29
        buf112 = buf93; del buf93  # reuse
        buf114 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf108, mul_905, buf112, buf114, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_905
        buf113 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf112, buf113, 128, 38, grid=grid(128), stream=stream0)
        buf115 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf114, buf115, 128, 38, grid=grid(128), stream=stream0)
        buf118 = reinterpret_tensor(buf86, (128, 512), (512, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (128, s0*s1), (1, 128), 0), view_28, out=buf118)
        del view_28
        buf119 = reinterpret_tensor(buf85, (s0*s1, 512), (512, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf117, permute_115, out=buf119)
        del permute_115
        buf120 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf118, buf120, 65536, grid=grid(65536), stream=stream0)
        buf121 = reinterpret_tensor(buf119, (s0, s1, 512), (512*s1, 512, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [gelu_2], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf121, mm_10, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_10
        buf122 = reinterpret_tensor(buf118, (512, 128), (128, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, s0*s1), (1, 512), 0), view_26, out=buf122)
        del view_26
        buf123 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (s0*s1, 512), (512, 1), 0), permute_119, out=buf123)
        del permute_119
        buf124 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf122, buf124, 65536, grid=grid(65536), stream=stream0)
        buf131 = buf116; del buf116  # reuse
        buf132 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf131, buf123, primals_25, mul_843, div_19, buf132, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_19
        del primals_25
        buf127 = buf114; del buf114  # reuse
        buf129 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf123, mul_843, buf127, buf129, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_843
        buf128 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf127, buf128, 128, 38, grid=grid(128), stream=stream0)
        buf130 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf129, buf130, 128, 38, grid=grid(128), stream=stream0)
        buf133 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_37, (s0*s1, 128), (128, 1), 0), out=buf133)
        buf134 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf132, permute_123, out=buf134)
        del buf132
        del permute_123
        buf135 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf133, buf135, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf136 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf134, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat_4, cat_5, permute_19, getitem_37, getitem_38, None, None, s1, s1, 0.0, True, getitem_43, getitem_44, scale=0.08838834764831843)
        del cat_4
        del cat_5
        del getitem_37
        del getitem_38
        del getitem_43
        del getitem_44
        del permute_19
        buf137 = buf136[0]
        buf138 = buf136[1]
        buf139 = buf136[2]
        del buf136
        buf140 = reinterpret_tensor(buf134, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf138, buf140, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf141 = reinterpret_tensor(buf138, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf137, buf141, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf137
        buf142 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf141, buf140, buf139, buf142, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        del buf139
        buf143 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (384, s0*s1), (1, 384), 0), view_20, out=buf143)
        del view_20
        buf144 = reinterpret_tensor(buf141, (s0*s1, 128), (128, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (s0*s1, 384), (384, 1), 0), permute_131, out=buf144)
        del permute_131
        buf145 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf143, buf145, 49152, grid=grid(49152), stream=stream0)
        buf152 = buf131; del buf131  # reuse
        buf153 = reinterpret_tensor(buf140, (s0*s1, 128), (128, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf152, buf144, primals_21, mul_606, div_20, buf153, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_20
        del primals_21
        buf148 = buf129; del buf129  # reuse
        buf150 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf144, mul_606, buf148, buf150, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_606
        buf149 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf148, buf149, 128, 38, grid=grid(128), stream=stream0)
        buf151 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf150, buf151, 128, 38, grid=grid(128), stream=stream0)
        buf154 = reinterpret_tensor(buf122, (128, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (128, s0*s1), (1, 128), 0), view_18, out=buf154)
        del view_18
        buf155 = reinterpret_tensor(buf121, (s0*s1, 512), (512, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf153, permute_135, out=buf155)
        del permute_135
        buf156 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf154, buf156, 65536, grid=grid(65536), stream=stream0)
        buf157 = reinterpret_tensor(buf155, (s0, s1, 512), (512*s1, 512, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [gelu_1], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf157, mm_6, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_6
        buf158 = reinterpret_tensor(buf154, (512, 128), (128, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (512, s0*s1), (1, 512), 0), view_16, out=buf158)
        del view_16
        buf159 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (s0*s1, 512), (512, 1), 0), permute_139, out=buf159)
        del permute_139
        buf160 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf158, buf160, 65536, grid=grid(65536), stream=stream0)
        buf167 = buf152; del buf152  # reuse
        buf168 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf167, buf159, primals_17, mul_544, div_21, buf168, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_21
        del primals_17
        buf163 = buf150; del buf150  # reuse
        buf165 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf159, mul_544, buf163, buf165, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_544
        buf164 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf163, buf164, 128, 38, grid=grid(128), stream=stream0)
        buf166 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf165, buf166, 128, 38, grid=grid(128), stream=stream0)
        buf169 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_21, (s0*s1, 128), (128, 1), 0), out=buf169)
        buf170 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf168, permute_143, out=buf170)
        del buf168
        del permute_143
        buf171 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf169, buf171, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf172 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf170, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat_2, cat_3, permute_11, getitem_21, getitem_22, None, None, s1, s1, 0.0, True, getitem_27, getitem_28, scale=0.08838834764831843)
        del cat_2
        del cat_3
        del getitem_21
        del getitem_22
        del getitem_27
        del getitem_28
        del permute_11
        buf173 = buf172[0]
        buf174 = buf172[1]
        buf175 = buf172[2]
        del buf172
        buf176 = reinterpret_tensor(buf170, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf174, buf176, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf177 = reinterpret_tensor(buf174, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf173, buf177, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf173
        buf178 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf177, buf176, buf175, buf178, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        buf179 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (384, s0*s1), (1, 384), 0), view_10, out=buf179)
        del view_10
        buf180 = reinterpret_tensor(buf177, (s0*s1, 128), (128, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (s0*s1, 384), (384, 1), 0), permute_151, out=buf180)
        del permute_151
        buf181 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf179, buf181, 49152, grid=grid(49152), stream=stream0)
        buf188 = buf167; del buf167  # reuse
        buf189 = reinterpret_tensor(buf176, (s0*s1, 128), (128, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_backward_view_5.run(buf188, buf180, primals_13, mul_307, div_22, buf189, triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_backward_view_5_xnumel), stream=stream0)
        del div_22
        del primals_13
        buf184 = buf165; del buf165  # reuse
        buf186 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_1.run(buf180, mul_307, buf184, buf186, s0, s1, 4864, triton_red_fused__to_copy_native_layer_norm_backward_1_rnumel, grid=grid(4864), stream=stream0)
        del mul_307
        buf185 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf184, buf185, 128, 38, grid=grid(128), stream=stream0)
        buf187 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf186, buf187, 128, 38, grid=grid(128), stream=stream0)
        buf190 = reinterpret_tensor(buf158, (128, 512), (512, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (128, s0*s1), (1, 128), 0), view_8, out=buf190)
        del view_8
        buf191 = reinterpret_tensor(buf157, (s0*s1, 512), (512, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, permute_155, out=buf191)
        del permute_155
        buf192 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf190, buf192, 65536, grid=grid(65536), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (s0, s1, 512), (512*s1, 512, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu_backward, aten.gelu]
        triton_poi_fused_gelu_gelu_backward_4_xnumel = 512*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_4.run(buf193, mm_2, triton_poi_fused_gelu_gelu_backward_4_xnumel, grid=grid(triton_poi_fused_gelu_gelu_backward_4_xnumel), stream=stream0)
        del mm_2
        buf194 = reinterpret_tensor(buf190, (512, 128), (128, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (512, s0*s1), (1, 512), 0), view_6, out=buf194)
        del view_6
        buf195 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (s0*s1, 512), (512, 1), 0), permute_159, out=buf195)
        del buf193
        del permute_159
        buf196 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(buf194, buf196, 65536, grid=grid(65536), stream=stream0)
        del buf194
        buf203 = buf188; del buf188  # reuse
        buf204 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_1, layer_norm_1], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.view]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf203, buf195, primals_9, embedding, mm_1, getitem_15, rsqrt_1, buf204, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_10_xnumel), stream=stream0)
        del primals_9
        buf206 = reinterpret_tensor(buf175, (s0*s1, 128), (128, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf204, permute_163, out=buf206)
        del permute_163
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf208 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf206, (s0, 1, s1, 128), (128*s1, 128, 128, 1), 0), cat, cat_1, permute_3, getitem_5, getitem_6, None, None, s1, s1, 0.0, True, getitem_11, getitem_12, scale=0.08838834764831843)
        del cat
        del cat_1
        del getitem_11
        del getitem_12
        del getitem_6
        del permute_3
        buf211 = buf208[2]
        buf210 = buf208[1]
        buf212 = reinterpret_tensor(buf206, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf210, buf212, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        buf209 = buf208[0]
        del buf208
        buf213 = reinterpret_tensor(buf210, (s0, 1, s1, 128), (128*s1, 128*s0*s1, 128, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [angles, cos, cos_1, sin, sin_1], Original ATen: [aten.mul, aten.cos, aten._to_copy, aten.sin, aten.slice_backward, aten.neg, aten.add]
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel = 128*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7.run(buf209, buf213, s1, triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel, grid=grid(triton_poi_fused__to_copy_add_cos_mul_neg_sin_slice_backward_7_xnumel), stream=stream0)
        del buf209
        buf214 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_8_xnumel = 384*s0*s1
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf213, buf212, buf211, buf214, triton_poi_fused_stack_8_xnumel, grid=grid(triton_poi_fused_stack_8_xnumel), stream=stream0)
        del buf211
        del buf212
        buf216 = reinterpret_tensor(buf213, (s0*s1, 128), (128, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (s0*s1, 384), (384, 1), 0), permute_171, out=buf216)
        del permute_171
        buf199 = buf186; del buf186  # reuse
        buf220 = buf184; del buf184  # reuse
        buf201 = empty_strided_cuda((128, 38), (1, 128), torch.float32)
        buf222 = empty_strided_cuda((128, 38), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, layer_norm_1, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_rnumel = (37 + s0*s1) // 38
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11.run(buf195, embedding, mm_1, getitem_15, rsqrt_1, buf216, getitem_1, rsqrt, buf199, buf220, buf201, buf222, s0, s1, 4864, triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_11_rnumel, grid=grid(4864), stream=stream0)
        del buf195
        del getitem_15
        del mm_1
        del rsqrt_1
        buf200 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, layer_norm_1], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf199, buf200, 128, 38, grid=grid(128), stream=stream0)
        del buf199
        buf202 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf201, buf202, 128, 38, grid=grid(128), stream=stream0)
        del buf201
        buf205 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (128, s0*s1), (1, 128), 0), reinterpret_tensor(getitem_5, (s0*s1, 128), (128, 1), 0), out=buf205)
        del buf204
        del getitem_5
        buf207 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf205, buf207, 16384, grid=grid(16384), stream=stream0)
        del buf205
        buf215 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (384, s0*s1), (1, 384), 0), view, out=buf215)
        del buf214
        del view
        buf217 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf215, buf217, 49152, grid=grid(49152), stream=stream0)
        del buf215
        buf225 = empty_strided_cuda((270, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_12.run(buf225, 34560, grid=grid(34560), stream=stream0)
        buf224 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.embedding_dense_backward]
        triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13_xnumel = s0*s1
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13.run(buf224, buf216, primals_5, embedding, getitem_1, rsqrt, primals_3, buf225, triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13_xnumel, 128, grid=grid(triton_per_fused__to_copy_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_13_xnumel), stream=stream0)
        del buf216
        del buf224
        del embedding
        del getitem_1
        del primals_3
        del primals_5
        del rsqrt
        buf221 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf220, buf221, 128, 38, grid=grid(128), stream=stream0)
        del buf220
        buf223 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_backward_2.run(buf222, buf223, 128, 38, grid=grid(128), stream=stream0)
        del buf222
        buf227 = empty_strided_cuda((270, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_14.run(buf0, buf225, buf227, 34560, grid=grid(34560), stream=stream0)
        del buf0
        del buf225
    return (None, None, None, buf227, buf221, buf223, buf217, buf207, buf200, buf202, buf196, buf192, buf185, buf187, buf181, buf171, buf164, buf166, buf160, buf156, buf149, buf151, buf145, buf135, buf128, buf130, buf124, buf120, buf113, buf115, buf109, buf99, buf92, buf94, buf88, buf84, buf77, buf79, buf73, buf63, buf56, buf58, buf52, buf48, buf41, buf43, buf37, buf27, buf20, buf22, buf16, buf12, buf6, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 64
    primals_2 = 76
    mul_20 = 4864
    primals_3 = rand_strided((64, 76), (76, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    embedding = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_1 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_3 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_6 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mm_1 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_15 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_2 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_8 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_307 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_10 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_2 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_3 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_11 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_22 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_28 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_544 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_6 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_18 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_606 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_4 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_5 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_19 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_38 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_44 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_843 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_10 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_28 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_905 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_6 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_7 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_27 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_54 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_1142 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_14 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_38 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_1204 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_8 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_9 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_35 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_69 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_70 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_76 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_1441 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_48 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_1503 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_10 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_11 = rand_strided((64, 1, 76, 128), (9728, 9728, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_43 = rand_strided((64, 1, 76, 128), (29184, 128, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_85 = rand_strided((64, 1, 76, 128), (9728, 128, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_86 = rand_strided((64, 1, 76), (76, 76, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_92 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mul_1740 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_58 = rand_strided((4864, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_1802 = rand_strided((64, 76, 128), (9728, 128, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((4864, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_51 = rand_strided((270, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_12 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_55 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_59 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_13 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_63 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_71 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_14 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_79 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_15 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_83 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_91 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_16 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_99 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_17 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_111 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_18 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_119 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_19 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_131 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_20 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_139 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_21 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_151 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    div_22 = rand_strided((64, 76, 1), (76, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_159 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_163 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_171 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((64, 76, 270), (20520, 270, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, mul_20, primals_3, primals_5, primals_9, primals_13, primals_17, primals_21, primals_25, primals_29, primals_33, primals_37, primals_41, primals_45, primals_49, primals_53, embedding, getitem_1, rsqrt, view, cat, cat_1, permute_3, getitem_5, getitem_6, getitem_11, getitem_12, mm_1, getitem_15, rsqrt_1, view_6, mm_2, view_8, mul_307, view_10, cat_2, cat_3, permute_11, getitem_21, getitem_22, getitem_27, getitem_28, mul_544, view_16, mm_6, view_18, mul_606, view_20, cat_4, cat_5, permute_19, getitem_37, getitem_38, getitem_43, getitem_44, mul_843, view_26, mm_10, view_28, mul_905, view_30, cat_6, cat_7, permute_27, getitem_53, getitem_54, getitem_59, getitem_60, mul_1142, view_36, mm_14, view_38, mul_1204, view_40, cat_8, cat_9, permute_35, getitem_69, getitem_70, getitem_75, getitem_76, mul_1441, view_46, mm_18, view_48, mul_1503, view_50, cat_10, cat_11, permute_43, getitem_85, getitem_86, getitem_91, getitem_92, mul_1740, view_56, mm_22, view_58, mul_1802, view_60, permute_51, div_12, permute_55, permute_59, div_13, permute_63, permute_71, div_14, permute_75, permute_79, div_15, permute_83, permute_91, div_16, permute_95, permute_99, div_17, permute_103, permute_111, div_18, permute_115, permute_119, div_19, permute_123, permute_131, div_20, permute_135, permute_139, div_21, permute_143, permute_151, div_22, permute_155, permute_159, permute_163, permute_171, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
