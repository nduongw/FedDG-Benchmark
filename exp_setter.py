import json
import os

class ExpSetter(object):
    def __init__(self, algorithm) -> None:
        pass
        

    

class Exp1(ExpSetter):
    def __init__(self, algorithm) -> None:
        self.config_file = f"./config/{algorithm}/config_baseline.json"
        with open(self.config_file) as f:
            self.config = json.load(f)
        self.target_path = f"./config/{algorithm}/{self.__class__.__name__}/"
        os.makedirs(self.target_path)
    
    def set(self):
        exp_num = 0
        dataset_list = ["IWildCam"]
        num_clients_list = [243]
        self.config['server']['criterion'] = "torch.nn.BCELoss"

        iid_list = [0, 1]
        lr_list = [5e-5]
        for i, dataset in enumerate(dataset_list):
            num_clients = num_clients_list[i]
            self.config['global']["dataset_name"] = dataset
            self.config['global']['num_clients'] = num_clients
            lr = lr_list[i]
            self.config['client']['optimizer_config']['lr'] = lr
            
            # for j, test_domain in enumerate(test_domain_list[i]):
            #     self.config['global']['test_domain'] = test_domain
            for iid in iid_list:

                self.config['global']['iid'] = iid
                with open(os.path.join(self.target_path, f"config_{exp_num}.json"), "w") as f:
                    json.dump(self.config, f, indent=4)
                exp_num += 1
        return True

# if __name__ == "__main__":
#     exp = Exp1("ERM")


