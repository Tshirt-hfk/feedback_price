

class Arguments:

    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])
