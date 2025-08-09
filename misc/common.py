import importlib


def get_class(_module: str, class_name: str):
    _module = importlib.import_module(_module, package=None)
    return getattr(_module, class_name)


def make_object(module: str, class_name: str, args=None):
    _class = get_class(module, class_name)
    return _class() if args is None else _class(**args)


def make_object_from_config(config):
    if isinstance(config, list):
        return [make_object_from_config(i) for i in config]
    elif isinstance(config, dict):
        if 'module' in config and 'class_name' in config:
            if 'args' in config:
                config['args'] = make_object_from_config(config['args'])

            return make_object(**config)
        else:
            return {
                k: make_object_from_config(v)
                for k, v in config.items()
            }
    else:
        return config