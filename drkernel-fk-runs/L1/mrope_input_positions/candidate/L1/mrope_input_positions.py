import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _compute_mrope_positions_kernel(
    out_ptr,            # *int64, length = 3*L
    start_ptr,          # *int64, length = K
    length_ptr,         # *int64, length = K
    block_kind_ptr,     # *int32, length = K  (0=text,1=image,2=video)
    video_id_ptr,       # *int32, length = K  (valid if kind==2)
    per_frame,          # int32  (0 or 1)
    cumsum_video_offsets_ptr,  # *int64, length = NV+1
    video_grid_thw_ptr,  # *int32, length = 3*NV  (flattened T,H,W)
    video_second_per_grid_ptr, # *float32, length = NV
    image_grid_thw_ptr,  # *int32, length = 3*NI  (flattened T,H,W)
    num_images,          # int32
    spatial_merge_size,  # int32
    tokens_per_second,   # float32
    K,                   # int32  number of blocks
    L,                   # int32  total tokens
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L

    # Find, for each lane, the block that contains token k = offs
    block_id = tl.zeros([BLOCK], dtype=tl.int32)
    inblock = tl.zeros([BLOCK], dtype=tl.int32)

    # Sequential O(K) scan of block starts
    for bi in range(0, K):
        start_b = tl.load(start_ptr + bi)   # int64
        len_b = tl.load(length_ptr + bi)    # int64
        sofar = start_b + len_b             # int64

        inb = (offs >= start_b) & (offs < sofar) & mask  # bool
        # Update only lanes that are still unset and in-block
        cond = inb & (block_id == 0)
        block_id = tl.where(cond, tl.full([BLOCK], bi, dtype=tl.int32), block_id)
        inblock = tl.where(cond, (offs - start_b).to(tl.int32), inblock)

    # Load block kind for each lane
    kind_val = tl.load(block_kind_ptr + block_id)  # int32
    is_video_block = (kind_val == 2)
    # Video id
    vID = tl.load(video_id_ptr + block_id, mask=is_video_block, other=0)  # int32

    # Compute pos (int64) for each lane
    pos = tl.zeros([BLOCK], dtype=tl.int64)

    # per-frame or contiguous
    is_perframe = (per_frame != 0)

    # Load video params: T,H,W at index vID
    T = tl.load(video_grid_thw_ptr + 3 * vID + 0).to(tl.int32)
    H = tl.load(video_grid_thw_ptr + 3 * vID + 1).to(tl.int32)
    W = tl.load(video_grid_thw_ptr + 3 * vID + 2).to(tl.int32)
    h_size = (H // spatial_merge_size).to(tl.int32)
    w_size = (W // spatial_merge_size).to(tl.int32)

    # Second per grid factor (float), will be multiplied by tokens_per_second on host side if needed
    sec_grid = tl.load(video_second_per_grid_ptr + vID).to(tl.float32)

    if is_perframe:
        tokens_per_frame = (T * h_size * w_size).to(tl.int32)
        local = inblock  # int32
        frame_id = (local // tokens_per_frame).to(tl.int32)
        rem = (local - frame_id * tokens_per_frame).to(tl.int32)
        t = frame_id
        area = h_size * w_size
        h = (rem // area).to(tl.int32)
        w = (rem % area).to(tl.int32)
        # Compute pos_v as int64: t*factor + h*h_size + w
        # factor was pre-scaled on host if needed; here sec_grid is seconds per grid
        # We need to apply tokens_per_second on host side to get factor in token units.
        # But we passed sec_grid as-is; on host we multiplied tokens_per_second -> factor.
        # To keep host simple, we instead pass factor pre-scaled: factor = sec_grid * tokens_per_second
        # However, in this kernel branch we don't have that. So refactor: on host,set video_second_per_grid to factor.
        # Alternative: compute here but we don't have tokens_per_second. -> Simplify: on host,set video_second_per_grid to factor.
        # I'll implement that: pass video_second_per_grid as factor already.
        # Then: pos_v = t*video_second_per_grid + h*h_size + w  but video_second_per_grid is factor (tokens), not seconds.
        # So: pos_v = t*factor + h*h_size + w
        # factor is tokens, so multiply by 1.0 is OK; but we have float. Cast to int64.
        # To avoid fp, on host set video_second_per_grid to factor (already multiplied by tokens_per_second).
        # I will assume that; if not, default to 0.
        factor_tokens = sec_grid  # but see above: we need tokens, not seconds.
        # Given possible confusion, I'll instead compute using int64 directly without float.
        # Reroute: use int math only.
        # But factor is in tokens -> OK as float, then cast.
        t64 = t.to(tl.int64)
        h64 = h.to(tl.int64)
        w64 = w.to(tl.int64)
        fac64 = (sec_grid * tokens_per_second).to(tl.int64)  # on host side tokens_per_second converts sec->tok
        pos_v = t64 * fac64 + h64 * h_size.to(tl.int64) + w64
        pos = tl.where(is_video_block, pos_v, pos)
    else:
        total = (T * h_size * w_size).to(tl.int32)
        local = inblock  # int32
        frame_id = (local // total).to(tl.int32)
        rem = (local - frame_id * total).to(tl.int32)
        t = frame_id
        area = h_size * w_size
        h = (rem // area).to(tl.int32)
        w = (rem % area).to(tl.int32)
        t64 = t.to(tl.int64)
        h64 = h.to(tl.int64)
        w64 = w.to(tl.int64)
        fac64 = (sec_grid * tokens_per_second).to(tl.int64)
        pos_v = t64 * fac64 + h64 * h_size.to(tl.int64) + w64
        pos = tl.where(is_video_block, pos_v, pos)

    # For image or text blocks: pos = inblock (int64)
    pos = tl.where((kind_val == 0) | (kind_val == 1), inblock.to(tl.int64), pos)

    # Finally, out value = start[block_id] + pos
    start_b_selected = tl.load(start_ptr + block_id)  # int64
    out_val = start_b_selected + pos  # int64

    # Store to out: shape (3, L) -> linear as 3*L
    base = offs * 3
    tl.store(out_ptr + base + 0, out_val, mask=mask)
    tl.store(out_ptr + base + 1, out_val, mask=mask)
    tl.store(out_ptr + base + 2, out_val, mask=mask)


class ModelNew(nn.Module):
    """Triton-optimized version of Model that computes M-RoPE 3D positions.

    Same __init__ and forward signature as original Model.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_tokens: list[int],
        spatial_merge_size: int,
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
        video_second_per_grid: list[float] | None = None,
        tokens_per_second: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        # Fallback to CPU / non-CUDA: use original Model
        if not torch.cuda.is_available():
            m = Model()
            return m.forward(
                input_tokens,
                spatial_merge_size,
                image_grid_thw,
                video_grid_thw,
                image_offsets,
                video_offsets,
                video_second_per_grid,
                tokens_per_second,
            )

        device = torch.device("cuda")
        L = len(input_tokens)

        # 1) Build block metadata on host: K blocks total
        blocks = []
        st = 0

        # Text prefix
        if st < L:
            blocks.append((st, L - st, 0))  # (start, length, kind=0)
            st = L

        num_images = 0
        if image_grid_thw is not None and image_offsets is not None:
            for i, (t, h, w) in enumerate(image_grid_thw):
                merged_h = max(1, h // spatial_merge_size)
                merged_w = max(1, w // spatial_merge_size)
                grid_size = t * merged_h * merged_w
                start = st
                length = grid_size
                kind = 1
                blocks.append((start, length, kind))
                st += length
                num_images += 1

        num_videos = 0
        per_frame = 0  # 0 => contiguous per video, 1 => per-frame offsets
        if video_grid_thw is not None and video_offsets is not None:
            total_frames = sum(thw[0] for thw in video_grid_thw)
            per_frame = 1 if (len(video_offsets) == total_frames and total_frames > len(video_grid_thw)) else 0
            for i, (t, h, w) in enumerate(video_grid_thw):
                merged_h = max(1, h // spatial_merge_size)
                merged_w = max(1, w // spatial_merge_size)
                if per_frame:
                    for _ in range(t):
                        start = st
                        length = merged_h * merged_w
                        kind = 2
                        blocks.append((start, length, kind))
                        st += length
                    num_videos += 1
                else:
                    start = st
                    length = t * merged_h * merged_w
                    kind = 2
                    blocks.append((start, length, kind))
                    st += length
                    num_videos += 1

        # Trailing text
        if st < L:
            start = st
            length = L - st
            kind = 0
            blocks.append((start, length, kind))

        K = len(blocks)

        # If K is too large for our simple block search, fall back
        MAX_BLOCK_SEARCH = 1024
        if K > MAX_BLOCK_SEARCH:
            m = Model()
            return m.forward(
                input_tokens,
                spatial_merge_size,
                image_grid_thw,
                video_grid_thw,
                image_offsets,
                video_offsets,
                video_second_per_grid,
                tokens_per_second,
            )

        # 2) Construct start, length, kind arrays
        start = torch.tensor([b[0] for b in blocks], dtype=torch.int64, device=device)
        length = torch.tensor([b[1] for b in blocks], dtype=torch.int64, device=device)
        block_kind = torch.tensor([b[2] for b in blocks], dtype=torch.int32, device=device)

        # 3) Video IDs array (only for kind==2 blocks)
        video_id = torch.zeros(K, dtype=torch.int32, device=device)
        for bi, k in enumerate(block_kind.cpu().numpy()):
            if k == 2:
                video_id[bi] = int(bi - num_images)

        # 4) Per-frame flag
        per_frame_t = int(per_frame)

        # 5) Cumsum video offsets (prefix sums)
        if video_offsets is not None:
            cumsum = [0]
            for o in video_offsets:
                cumsum.append(cumsum[-1] + o)
            cumsum_video_offsets = torch.tensor(cumsum, dtype=torch.int64, device=device)
        else:
            cumsum_video_offsets = torch.tensor([0, 0], dtype=torch.int64, device=device)

        # 6) Flatten video_grid_thw to int array [3*NV]
        if video_grid_thw is not None:
            flat = []
            for t, h, w in video_grid_thw:
                flat.extend([t, h, w])
            video_grid_thw_t = torch.tensor(flat, dtype=torch.int32, device=device)
        else:
            video_grid_thw_t = torch.empty(0, dtype=torch.int32, device=device)

        # 7) video_second_per_grid: pass as factor = second_per_grid * tokens_per_second  (tokens units)
        if video_second_per_grid is not None:
            video_factor = [float(s) * float(tokens_per_second) for s in video_second_per_grid]
            video_second_per_grid_t = torch.tensor(video_factor, dtype=torch.float32, device=device)
        else:
            video_second_per_grid_t = torch.tensor([1.0], dtype=torch.float32, device=device)

        # 8) image_grid_thw flatten
        if image_grid_thw is not None:
            flat = []
            for t, h, w in image_grid_thw:
                flat.extend([t, h, w])
            image_grid_thw_t = torch.tensor(flat, dtype=torch.int32, device=device)
        else:
            image_grid_thw_t = torch.empty(0, dtype=torch.int32, device=device)

        # 9) Allocate output
        out = torch.empty((3, L), dtype=torch.int64, device=device)
        out_ptr = out.reshape(-1)

        # 10) Launch kernel
        BLOCK = 1024
        grid = (triton.cdiv(L, BLOCK),)
        _compute_mrope_positions_kernel[grid](
            out_ptr,
            start,
            length,
            block_kind,
            video_id,
            per_frame_t,
            cumsum_video_offsets,   # not used in kernel but kept for API symmetry
            video_grid_thw_t,
            video_second_per_grid_t,  # this now carries factor (tokens), not seconds
            image_grid_thw_t,         # not used in kernel but kept
            int(num_images),
            int(spatial_merge_size),
            float(tokens_per_second),  # not used in kernel but kept
            int(K),
            int(L),
            BLOCK=BLOCK,
        )

        # 11) Compute mrope_position_delta on device
        maxv = int(torch.max(out).item())
        delta = maxv - L

        return out, delta

MRopeInputPositions = ModelNew
