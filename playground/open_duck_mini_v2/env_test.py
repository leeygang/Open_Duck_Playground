import jax, jax.numpy as jp
from playground.open_duck_mini_v2 import standing

env = standing.Standing(task="duck_terrain")
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
state = env.reset(reset_rng)

# print("reset ok. obs shape:", state.obs["state"].shape)
# for i in range(20):
#     rng, act_rng = jax.random.split(rng)
#     # random action in [-1,1] matching action_size
#     action = jax.random.uniform(act_rng, (env.action_size,), minval=-1.0, maxval=1.0)
#     rng, step_rng = jax.random.split(rng)
#     state.info["rng"] = step_rng
#     state = env.step(state, action)
#     print(f"step {i:02d}: reward={state.reward} done={bool(state.done)}")
print("done smoke test")