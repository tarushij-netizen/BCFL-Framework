import csv
import logging
import os
from copy import deepcopy
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from config.algorithm import Algorithm
from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client, BaseClient, SignClient
from client.trainer.fedproxTrainer import fedproxTrainer
from client.trainer.SignTrainer import SignTrainer
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
from chainfl.interact import chain_proxy


class Task:
    '''
    WorkFlow of a Task:
    0. Construct (Model, Dataset)--> Benchmark
    1. Construct (Server, Client)--> FL Algorithm
    3. Process of the dataset 
    '''
    def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
        self.global_args = global_args
        self.train_args = train_args
        self.model = None
        
        #Get Dataset
        # TODO pass the schema (object) instead of args directly.  
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'),True)
        self.test_dataset =  DatasetFactory().get_dataset(global_args.get('dataset'),False)
        #Get Model
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #FL alg
        logger.info("Algorithm: {algorithm}")
        self.server = algorithm.get_server()
        self.server = self.server()
        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()
        
        #Get Client and Trainer
        self.client_list = None
        self.client_pool : list[Client] = []
        
    def __repr__(self) -> str:
        pass
    
    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
                    , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        if self.global_args.get('non-iid'):
            # Dirichlet split: smaller alpha = more skewed client data
            self.train_dataloader_list = DatasetSpliter().dirichlet_split(dataset     = self.train_dataset,
                                                                          client_list = chain_proxy.get_client_list(),
                                                                          batch_size  = batch_size,
                                                                          alpha       = self.global_args.get('alpha', 1))
        else:
            self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
                                                                       client_list = chain_proxy.get_client_list(),
                                                                       batch_size  = batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def _construct_sign(self):
        self.keys_dict = dict()
        self.keys = list()
        sign_num = self.global_args.get('sign_num')
        if(None == sign_num): 
            sign_num = 0
            logger.info("No client need to add watermark")
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = None
        else:
            logger.info(f"{sign_num} client(s) will inject watermark into their models")
            
            for i in range(self.global_args.get('client_num')):
                if i < self.global_args.get('sign_num'):
                    key = chain_proxy.construct_sign(self.global_args)
                    self.keys.append(key)
                else : 
                    self.keys.append(None)
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = self.keys[ind]
            #Project the watermake to the client TODO work with the blockchain
            #Get model Here better split another function.                 
            tmp_args = chain_proxy.construct_sign(self.global_args)
            self.model = ModelFactory().get_sign_model(model          = self.global_args.get('model'),
                                                       class_num      = self.global_args.get('class_num'),
                                                       in_channels    = self.global_args.get('in_channels'),
                                                       watermark_args = tmp_args)  
        return    
    
    def _regist_client(self):
        #Regist the client to the blockchain.
        for i in range(self.global_args['client_num']):
            chain_proxy.client_regist()
        self.client_list = chain_proxy.get_client_list()
    
    def _construct_client(self):
        # Data-poisoning attacks (labelflip / backdoor): malicious clients train on poisoned data.
        attack = getattr(self.server, 'attack', 'none')
        malicious = set(getattr(self.server, 'malicious', []))
        data_attack = attack in ('labelflip', 'backdoor')
        if data_attack:
            from client.poisonedLoader import PoisonedLoader
        for ind, (client_id, _) in enumerate(self.client_list.items()):
            loader = self.train_dataloader_list[client_id]
            if data_attack and ind in malicious:
                loader = PoisonedLoader(loader, attack,
                                        num_classes=self.global_args.get('class_num', 10),
                                        target=self.global_args.get('backdoor_target', 0),
                                        poison_frac=self.global_args.get('poison_frac', 0.5))
            new_client = self.client(client_id, loader, self.model,
                                    self.trainer, self.train_args, self.test_dataloader, self.keys_dict[client_id])
            self.client_pool.append(new_client)
    
    def _evaluate_global(self, global_model) -> dict:
        """Accuracy/loss of the AGGREGATED global model on the full test set."""
        device = self.train_args.get('device', 'cpu')
        if self._eval_model is None:
            self._eval_model = deepcopy(self.client_pool[0].model)
        model = self._eval_model
        model.load_state_dict(global_model)
        model.to(device)
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        backdoor = getattr(self.server, 'attack', 'none') == 'backdoor'
        target = self.global_args.get('backdoor_target', 0)
        hit, cand = 0, 0
        if backdoor:
            from client.poisonedLoader import add_trigger
        with torch.no_grad():
            for data, targets in self.test_dataloader:
                data, targets = data.to(device), targets.to(device)
                out = model(data)
                loss_sum += torch.nn.functional.cross_entropy(out, targets, reduction='sum').item()
                correct += (out.argmax(1) == targets).sum().item()
                total += targets.size(0)
                if backdoor:
                    keep = targets != target
                    if keep.any():
                        pred = model(add_trigger(data[keep])).argmax(1)
                        hit += (pred == target).sum().item()
                        cand += int(keep.sum().item())
        res = {'acc': 100.0 * correct / total, 'loss': loss_sum / total}
        res['asr'] = 100.0 * hit / cand if backdoor and cand else None
        return res

    def run(self):
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_client()

        # Give the server the initial global model (needed by the attack
        # simulation and by similarityAggregator). Plain fedavgAggregator has no
        # such method, so the original baseline is unaffected.
        if hasattr(self.server, 'set_global_model'):
            self.server.set_global_model(self.client_pool[0].get_model_state_dict())

        self._eval_model = None
        malicious = set(getattr(self.server, 'malicious', []))
        # attack timing (always / onoff / sleeper), shared by data and model attacks
        from client.poisonedLoader import SCHEDULE, attack_on
        SCHEDULE['mode'] = self.global_args.get('attack_mode', 'always')
        SCHEDULE['period'] = self.global_args.get('attack_period', 5)
        whitewash = self.global_args.get('whitewash', False)
        rep_obj = getattr(self.server, 'reputation', None)
        if whitewash and rep_obj is None:
            raise ValueError("--whitewash needs --aggregator repsketch (there is no reputation to escape otherwise)")
        results_path = self.global_args.get('results_csv')
        ledger = None
        if self.global_args.get('onchain'):
            from chainfl.incentive import OnchainIncentives
            prefix = (results_path[:-4] if results_path and results_path.endswith('.csv') else 'results/run') + '_onchain'
            ledger = OnchainIncentives(len(self.client_pool), malicious=malicious,
                                       epoch_len=self.global_args.get('epoch_len', 5),
                                       init_rep=rep_obj.init if rep_obj is not None else 0.75,
                                       reward_tau=self.global_args.get('reward_tau', 0.9),
                                       out_prefix=prefix,
                                       sybil=self.global_args.get('sybil', 0))
        n_rejoins = 0
        att_rounds = 0          # rounds in which the attackers attacked
        att_included = 0        # attacker uploads that went into the model while attacking
        asr_list = []
        writer = None
        if results_path:
            os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
            f = open(results_path, 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['round', 'global_acc', 'global_loss', 'rejected',
                             'malicious_caught', 'honest_rejected', 'backdoor_asr',
                             'attack_on', 'malicious_included'])
        client_test = self.global_args.get('client_test', True)

        for i in range(self.global_args['communication_round']):
            SCHEDULE['round'] = i
            on = bool(malicious) and getattr(self.server, 'attack', 'none') != 'none' and attack_on(i)
            for client in self.client_pool:
                client.train(epoch = i)
                if client_test:
                    client.test(epoch = i)
                client.sign_test(epoch = i)
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            for client in self.client_pool:
                client.load_state_dict(global_model)

            res = self._evaluate_global(global_model)
            rejected = list(getattr(self.server, 'rejected_last_round', []))
            if ledger is not None:
                n = len(self.client_pool)
                hashes = getattr(self.server, 'last_update_hashes', None) or [None] * n
                rep = getattr(self.server, 'last_reputation', None) or None
                ledger.on_round(i, [c for c in range(n) if c not in rejected], hashes, rep)
            if whitewash:
                # an attacker that has been shut out drops its identity and comes back as a new one
                for m in sorted(malicious):
                    r_now = rep_obj.get(m)
                    if r_now < rep_obj.tau:
                        rep_obj.reset(m)
                        n_rejoins += 1
                        if ledger is not None:
                            ledger.rejoin(m, r_now)
                        print(f"[whitewash] round {i}: attacker {m} (reputation {r_now:.3f}) re-registers as a new identity")
            caught = len([r for r in rejected if r in malicious])
            honest_rej = len([r for r in rejected if r not in malicious])
            mal_in = len(malicious) - caught
            if on:
                att_rounds += 1
                att_included += mal_in
            if res.get('asr') is not None and i >= 3:  # rounds 0-2: untrained model, not an attack effect
                asr_list.append(res['asr'])
            logger.info(f"Global round {i}: test acc {res['acc']:.2f}, loss {res['loss']:.4f}, "
                        f"rejected {rejected}, malicious caught {caught}/{len(malicious)}, honest rejected {honest_rej}")
            asr = res.get('asr')
            asr_txt = f" | backdoor ASR {asr:6.2f}%" if asr is not None else ""
            on_txt = "" if self.global_args.get('attack_mode', 'always') == 'always' else (" | ATTACKING" if on else " | attackers honest")
            print(f"round {i:3d} | global test acc {res['acc']:6.2f}% | loss {res['loss']:.4f}{asr_txt} | "
                  f"rejected {rejected}{on_txt}", flush=True)
            if writer:
                writer.writerow([i, f"{res['acc']:.4f}", f"{res['loss']:.6f}",
                                 ' '.join(map(str, rejected)), caught, honest_rej,
                                 '' if asr is None else f"{asr:.4f}", int(on), mal_in])
                f.flush()
        if writer:
            f.close()
        extra = {
            "final_acc": round(res['acc'], 2),
            "final_asr": None if res.get('asr') is None else round(res['asr'], 2),
            "max_asr_from_round3": round(max(asr_list), 2) if asr_list else None,
            "attack_rounds": att_rounds,
            "attacker_uploads_included_while_attacking": att_included,
            "attacker_uploads_while_attacking": att_rounds * len(malicious),
            "whitewash_rejoins": n_rejoins,
        }
        print("[summary] " + ", ".join(f"{k} {v}" for k, v in extra.items()))
        if ledger is not None:
            ledger.finish(extra)
