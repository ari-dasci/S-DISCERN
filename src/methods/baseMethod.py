class BaseMethod:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def setup(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        return train_X, train_y, retain_X, retain_y, forget_X, forget_y

    def unlearn(self, train_X= None, train_y= None, retain_X= None, retain_y= None, forget_X= None, forget_y= None, transform_train= None, transform_retain= None, transform_forget = None):
        return self.model
