import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
    },
    {
        'id': 'S1-v1',
        'entry_point': ('diffuser.environments.s1:S1'),
    },
    {
        'id': 'T2-v1',
        'entry_point': ('diffuser.environments.manifolds:T2'),
    },
    {
        'id': 'T2-small-v1',
        'entry_point': ('diffuser.environments.manifolds:T2small'),
    },
    {
        'id': 'T2-medium-v1',
        'entry_point': ('diffuser.environments.manifolds:T2medium'),
    },
    {
        'id': 'T2-large-v1',
        'entry_point': ('diffuser.environments.manifolds:T2large'),
    },
    {
        'id': 'S2-small-v1',
        'entry_point': ('diffuser.environments.manifolds:S2small'),
    },
    {
        'id': 'S2-medium-v1',
        'entry_point': ('diffuser.environments.manifolds:S2medium'),
    },
    {
        'id': 'S2-large-v1',
        'entry_point': ('diffuser.environments.manifolds:S2large'),
    },
    {
        'id': 'SO3-small-v1',
        'entry_point': ('diffuser.environments.so3:SO3small'),
    },
    {
        'id': 'SO3-medium-v1',
        'entry_point': ('diffuser.environments.so3:SO3medium'),
    },
    {
        'id': 'SO3-large-v1',
        'entry_point': ('diffuser.environments.so3:SO3large'),
    },
    {
        'id': 'SO3GS-small-v1',
        'entry_point': ('diffuser.environments.so3:SO3GSsmall'),
    },
    {
        'id': 'SO3GS-medium-v1',
        'entry_point': ('diffuser.environments.so3:SO3GSmedium'),
    },
    {
        'id': 'SO3GS-large-v1',
        'entry_point': ('diffuser.environments.so3:SO3GSlarge'),
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()