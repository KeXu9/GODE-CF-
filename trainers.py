import numpy as np
import torch
import utils
from pprint import pprint
from utils import timer
from time import time
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, cal_mrr, get_user_performance_perpopularity, get_item_performance_perpopularity
import platform
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, args):

        self.args = args
        self.model = model
        self.dataset = dataset

        # Initialize device-specific attributes first
        self.is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'

        # Now we can safely call the optimizer method
        self.optim = self._get_optimized_optimizer()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.pred_mask_mat = self.dataset.valid_pred_mask_mat

        if self.is_apple_silicon:
            print("🍎 Apple Silicon optimizations enabled")

    def _get_optimized_optimizer(self):
        """Get optimized optimizer for M4 Mac"""
        if self.is_apple_silicon:
            # AdamW for Apple Silicon CPU
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'decay', 1e-4),
                eps=1e-8,
                amsgrad=True
            )
        else:
            return Adam(self.model.parameters(), lr=self.args.lr)

    def train(self, epoch):
        self.iteration(epoch)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, full_sort, mode='valid')

    def test(self, epoch, full_sort=False):
        self.pred_mask_mat = self.dataset.test_pred_mask_mat
        return self.iteration(epoch, full_sort, mode='test')

    def iteration(self, epoch, full_sort=False, mode='train'):
        raise NotImplementedError

    def complicated_eval(self):
        self.pred_mask_mat = self.dataset.test_pred_mask_mat
        return self.eval_analysis()

    def eval_analysis(self):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], 0
        recall_dict_list = []
        ndcg_dict_list = []
        for k in [1, 5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
            "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
            "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
            "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
            "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
            "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
            "MRR": '{:.8f}'.format(mrr)
        }
        print(post_fix, flush=True)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        """Save model state"""
        device_before = next(self.model.parameters()).device
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(device_before)

    def load(self, file_name):
        """Load model state"""
        if torch.cuda.is_available():
            map_location = 'cuda:0'
        else:
            map_location = 'cpu'

        state_dict = torch.load(file_name, map_location=map_location)
        self.model.load_state_dict(state_dict)


class GraphRecTrainer(Trainer):

    def __init__(self, model, dataset, args):
        super(GraphRecTrainer, self).__init__(
            model, dataset, args
        )

    def iteration(self, epoch, full_sort=False, mode='train'):

        if mode=='train':
            self.model.train()
            rec_avg_loss = 0.0
            avg_norm_loss = 0.0

            with timer(name="Sample"):
                S = utils.UniformSample_original(self.dataset)

            # Optimized tensor creation - create directly on device
            users = torch.from_numpy(S[:, 0]).long().to(self.args.device, non_blocking=True)
            posItems = torch.from_numpy(S[:, 1]).long().to(self.args.device, non_blocking=True)
            negItems = torch.from_numpy(S[:, 2]).long().to(self.args.device, non_blocking=True)

            users, posItems, negItems = utils.shuffle(users, posItems, negItems)
            total_batch = len(users) // self.args.bpr_batch + 1

            # Pre-allocate loss accumulation for better performance
            loss_accumulator = 0.0
            reg_loss_accumulator = 0.0

            # Create progress bar for batches
            batch_iterator = enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.bpr_batch))
            batch_pbar = tqdm(batch_iterator, total=total_batch, desc=f"🔥 Epoch {epoch+1}",
                             unit="batch", leave=False,
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            for (batch_i, (batch_users, batch_pos, batch_neg)) in batch_pbar:
                if self.args.model_name == 'UltraGCN':
                    loss = self.model(batch_users.cpu(), batch_pos.cpu(), batch_neg.cpu())

                    # Optimized gradient step
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()

                    # Efficient loss extraction without CPU transfer
                    loss_accumulator += loss.item()

                    # Update progress bar
                    batch_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{loss_accumulator/(batch_i+1):.4f}'
                    })
                else:
                    loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
                    reg_loss = reg_loss * self.args.decay
                    total_loss = loss + reg_loss

                    # Optimized gradient step
                    self.optim.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()

                    # Efficient loss extraction without redundant CPU transfers
                    loss_accumulator += loss.item()
                    reg_loss_accumulator += reg_loss.item()

                    # Update progress bar
                    batch_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Reg': f'{reg_loss.item():.4f}',
                        'Avg': f'{loss_accumulator/(batch_i+1):.4f}'
                    })

            # Close batch progress bar
            batch_pbar.close()

            # Final loss computation
            rec_avg_loss = loss_accumulator / total_batch
            if self.args.model_name != 'UltraGCN':
                avg_norm_loss = reg_loss_accumulator / total_batch

            # Memory cleanup after training epoch
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # Invalidate model cache for next epoch
            if hasattr(self.model, '_invalidate_cache'):
                self.model._invalidate_cache()


            time_info = timer.dict()
            timer.zero()
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss),
                "avg_norm_loss": '{:.4f}'.format(avg_norm_loss),
                "time_info": time_info
            }

            print(str(post_fix), flush=True)
        else:
            u_batch_size = self.args.testbatch
            if mode == 'valid':
                evalDict = self.dataset.validDict
            else:
                evalDict = self.dataset.testDict
                self.model.graph = self.dataset.getSparseGraph(self.dataset.test_pred_mask_mat)
            self.model.eval()

            pred_list = None

            with torch.no_grad():
                users = list(evalDict.keys())
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

                answer_list = None
                i = 0

                # Calculate total evaluation batches
                total_eval_batches = len(users) // u_batch_size + (1 if len(users) % u_batch_size != 0 else 0)

                # Create progress bar for evaluation
                eval_iterator = utils.minibatch(users, batch_size=u_batch_size)
                eval_pbar = tqdm(eval_iterator, total=total_eval_batches, desc=f"📊 Eval {mode}",
                               unit="batch", leave=False,
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

                for batch_users in eval_pbar:
                    groundTrue = [evalDict[u] for u in batch_users]

                    # Optimized tensor creation - direct to device
                    batch_users_gpu = torch.from_numpy(np.array(batch_users, dtype=np.int64)).to(self.args.device, non_blocking=True)

                    rating_pred = self.model.getUsersRating(batch_users_gpu)

                    # Optimized masking - keep on GPU longer
                    batch_user_index = batch_users_gpu.cpu().numpy().astype(np.int32)

                    # Validate indices before using them
                    if np.any(batch_user_index >= self.pred_mask_mat.shape[0]) or np.any(batch_user_index < 0):
                        print(f"⚠️ Invalid user indices detected: min={batch_user_index.min()}, max={batch_user_index.max()}, matrix shape={self.pred_mask_mat.shape}")
                        continue

                    # Convert to numpy only once
                    rating_pred_np = rating_pred.detach().cpu().numpy()

                    # Apply mask
                    rating_pred_np[self.pred_mask_mat[batch_user_index].toarray() > 0] = -(1<<10)

                    # Optimized top-k selection using torch.topk (faster)
                    rating_pred_tensor = torch.from_numpy(rating_pred_np)
                    _, batch_pred_list = torch.topk(rating_pred_tensor, k=40, dim=1)
                    batch_pred_list = batch_pred_list.numpy()

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = np.array(groundTrue)
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, np.array(groundTrue), axis=0)
                    i += 1

                    # Update evaluation progress
                    eval_pbar.set_postfix({
                        'Users': f'{i * u_batch_size}/{len(users)}',
                        'Batches': f'{i}/{total_eval_batches}'
                    })

                # Close evaluation progress bar
                eval_pbar.close()

                return self.get_full_sort_score(epoch, answer_list, pred_list)

    def eval_analysis(self):
        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = defaultdict(list)
        for user_id, item_id in zip(self.dataset.trainUser, self.dataset.trainItem):
            train[user_id].append(item_id)
            item_freq[item_id] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)

        u_batch_size = self.args.testbatch
        all_pos_items_ranks = defaultdict(list)
        testDict = self.dataset.testDict
        self.model.eval()

        pred_list = None

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

            answer_list = None
            i = 0
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.args.device)

                rating_pred = self.model.getUsersRating(batch_users_gpu)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = batch_users_gpu.cpu().numpy()
                rating_pred[self.pred_mask_mat[batch_user_index].toarray() > 0] = 0
                batch_pred_list = np.argsort(-rating_pred, axis=1)
                pos_items = np.array(groundTrue)

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = np.array(groundTrue)
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, np.array(groundTrue), axis=0)
                i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)
            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, self.dataset.m_items)
            return scores, result_info, None
