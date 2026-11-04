import sys

from torch.optim import AdamW, SGD

# In case method or config would change the optimizer we add them for posible future changes.
# As a 

def load_optimizer(model, method=None, config=None): 

    # Methods that always use AdamW regardless of backbone
    adamw_methods = ["NOISY", "DISTRIBUTIONAL", "PARAMETRIC", "SCRUB", "BADTEACHER", "RANDOMLABEL",
                     "BOUNDARYSHRINK", "BOUNDARYEXPAND", "UNSIR", "ISPF"]

    if config['pretrained_model'] == False:
        print("WARNING: Using random initialization instead of pretrained model. This may lead to worse performance and longer training times.")
        if model._base_model in ["vit_b_16", "swin_t"] or method in adamw_methods:
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=5e-2)

        elif model._base_model in ["efficientnet_v2_s"]:
            optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

        elif model._base_model in ["resnet18"] and method not in adamw_methods:
            optimizer = SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)

        else:
            print("ERROR UNRECOGNISED MODEL")
            sys.exit(-1)


    else:
        if model._base_model in ["vit_b_16", "swin_t"] or method in adamw_methods:
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=5e-2)

        elif model._base_model in ["efficientnet_v2_s"]:
            optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

        elif model._base_model in ["resnet18"] and method not in adamw_methods:
            optimizer = SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
        
        else:
            print("ERROR UNRECOGNISED MODEL")
            sys.exit(-1)
    return optimizer