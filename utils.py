def create_instance(instance_name, scope, *args, **kwargs):
    try:
        instance = scope[instance_name]
    except ImportError:
        raise ImportError(instance_name)
    except AttributeError:
        raise ImportError(instance_name)
    instance_functor = instance(*args, **kwargs)
    return instance_functor


def get_class(instance_name, scope):
    try:
        instance = scope[instance_name]
    except ImportError:
        raise ImportError(instance_name)
    except AttributeError:
        raise ImportError(instance_name)
    return instance
