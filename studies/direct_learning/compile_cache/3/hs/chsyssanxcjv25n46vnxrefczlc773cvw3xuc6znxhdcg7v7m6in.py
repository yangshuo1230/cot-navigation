
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
