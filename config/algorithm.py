import logging
from server.base.baseAggregator import ServerAggregator
from server.aggregation_alg.fedavg import fedavgAggregator
from client.base.baseTrainer import BaseTrainer
from client.trainer.normalTrainer import normalTrainer
from client.clients import Client, BaseClient
logger = logging.getLogger(__name__)

'''
FL Algorithm Info:
1. Write server derived from ServerAggregator
2. Write trainer derived from BaseTrainer
3. Write client(may not used) 
'''

class Algorithm:
    def __init__(self, 
                 server:  ServerAggregator  = None, 
                 trainer: BaseTrainer       = None,
                 client:  Client            = None):
        self.server  = server if server is not None else fedavgAggregator
        self.trainer = trainer if trainer is not None else normalTrainer
        self.client  = client if client is not None else BaseClient
    
    def get_server(self):
        return self.server
    
    def get_trainer(self):
        return self.trainer
    
    def get_client(self):
        return self.client
    

class FedAvg(Algorithm):
    def __init__(self):
        super(FedAvg, self).__init__()

from client.trainer.fedproxTrainer import fedproxTrainer
class FedProx(Algorithm):
    def __init__(self):
        super(FedProx, self).__init__(trainer=fedproxTrainer)

from client.trainer.SignTrainer import SignTrainer
from client.clients import SignClient  
class FedIPR(Algorithm):
    def __init__(self):
        super(FedIPR, self).__init__(trainer=SignTrainer, client=SignClient)
        
        

from server.aggregation_alg.robustAgg import make_server_class
class RobustFL(Algorithm):
    """FedAvg-style training with a chosen aggregator and optional simulated attackers.
    aggregator: fedavg | similarity | median | krum
    attack:     none | signflip | noise | freerider
    malicious:  0-based client indices whose uploads are poisoned
    """
    def __init__(self, aggregator="similarity", attack="none", malicious=(),
                 attack_scale=5.0, seed=0, agg_kwargs=None):
        server = make_server_class(aggregator, attack, malicious, attack_scale, seed, agg_kwargs)
        super(RobustFL, self).__init__(server=server)


__all__ = [
    "FedAvg",
    "FedProx",
    "FedIPR",
    "RobustFL",
]