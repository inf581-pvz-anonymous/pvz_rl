from gym.envs.registration import register

register(
    id='pvz-env-v0',
    entry_point='gym_pvz.envs:PVZEnv'
)

register(
    id='pvz-env-v1',
    entry_point='gym_pvz.envs:PVZEnv_V1'
)

register(
    id='pvz-env-v01',
    entry_point='gym_pvz.envs:PVZEnv_V01'
)

register(
    id='pvz-env-v2',
    entry_point='gym_pvz.envs:PVZEnv_V2'
)