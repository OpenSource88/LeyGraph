
class Sample_Par:
    def __init__(self, batch_size, dataset, sampler_setting, num=1):
        self.batch_size = batch_size
        self.dataset = dataset
        self.sampler_setting = sampler_setting
        self.num = num

