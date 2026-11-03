
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
