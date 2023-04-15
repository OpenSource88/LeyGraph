
class Model_Par:
    def __init__(self, model_type, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        self.model_type = model_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

