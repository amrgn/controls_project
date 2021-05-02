from gym.envs.registration import register

register(
    id='inverted-pend-v0',
    entry_point='gym_inverted_pend.envs:InvertedPend'
)

register(
    id='double-inverted-pend-v0',
    entry_point='gym_inverted_pend.envs:DoubleInvertedPend'
)