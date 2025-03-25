import triton
import triton.language as tl
import torch


@triton.jit
def pointer_jump_kernel(
    chum_ptr,  # [n] array of chum pointers
    n,  # number of nodes
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n
    my_chum = tl.load(chum_ptr + pid, mask=mask, other=-1)

    valid_mask = (my_chum != -1) & mask
    chums_chum = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    chums_chum = tl.where(
        valid_mask, tl.load(chum_ptr + my_chum, mask=valid_mask, other=-1), chums_chum
    )

    update_mask = valid_mask & (chums_chum != -1) & (chums_chum != my_chum)
    my_chum = tl.where(update_mask, chums_chum, my_chum)
    tl.store(chum_ptr + pid, my_chum, mask=mask)


def find_end_of_list(next_list):
    n = len(next_list)
    chum_tensor = torch.tensor(next_list, dtype=torch.int32, device="cuda")
    print("Initial chum_tensor:", chum_tensor)

    BLOCK_SIZE = 32
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_iterations = int(torch.log2(torch.tensor(float(n))).item()) + 1
    for i in range(max_iterations):
        pointer_jump_kernel[(grid,)](chum_tensor, n, BLOCK_SIZE=BLOCK_SIZE)
        print(f"After iteration {i}:", chum_tensor)

    return chum_tensor.cpu().numpy()


if __name__ == "__main__":
    # Create a linked list: 0->1->2->3->4->5->6->7->-1
    next_list = [1, 2, 3, 4, 5, 6, 7, -1]
    end_list = find_end_of_list(next_list)
    print("\nNode\tEnd of list from this node")
    for i, end in enumerate(end_list):
        print(f"{i}\t{end}")
