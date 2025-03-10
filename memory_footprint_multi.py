import torch

from torch.distributed._tools.mem_tracker import MemTracker
from activ_function_elusplus2L import Elusplus2L
from musonet import MUSONetRegressor

memory_footprints = {}
N = 1000
T = 3
n_features_per_source = 10

for Q in [3, 6, 9, 12, 15]:
    device, dtype = "cuda", torch.float32
    method = MUSONetRegressor(
        n_features=n_features_per_source * Q,
        n_hidden=32,
        n_shared=1,
        n_specific=2,
        learning_rate=0.0001,
        activation=Elusplus2L(),
        use_reduce_lr_on_plateau=True,
        early_stopping_patience=50,
        batch_size=200,
        max_epochs=50,
        device=device,
        verbose=0,
        n_jobs=1,
    )

    method._initialize()
    method = method.model

    device, dtype = "cuda", torch.float32
    optim = torch.optim.Adam(method.parameters(), foreach=True)
    mem_tracker = MemTracker()
    mem_tracker.track_external(method, optim)
    with mem_tracker as mt:
        for i in range(2):
            input_batch = torch.rand((N, n_features_per_source * Q), device=device, dtype=dtype)
            method(input_batch).sum().backward()
            optim.step()
            optim.zero_grad()
            if i == 0:
                # to account for lazy init of optimizer state
                mt.reset_mod_stats()
    mt.display_snapshot("peak", units="MiB", tabulate=True)
    mt.display_modulewise_snapshots(depth=2, units="MiB", tabulate=True)
    # Check for accuracy of peak memory
    # print(mt.get_tracker_snapshot("peak"))
    tracker_max = mt.get_tracker_snapshot("peak")[torch.device(type="cuda", index=0)]["Total"] / 1e6
    memory_footprints[Q] = tracker_max
    # cuda_max = torch.cuda.max_memory_allocated()
    # accuracy = tracker_max / cuda_max
    print(f"Q = {Q} → Max Memory Allocated: {tracker_max} MB")

print("------------------")
print(memory_footprints)
