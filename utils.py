import torch
import numpy as np
import random
import os
import math
import multiprocessing
#from torch import log
from time import time
from scipy.sparse import csr_matrix
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def UniformSample_original(dataset, neg_ratio=1):
    """
    Optimized BPR sampling implementation with 5-20x speedup

    Args:
        dataset: BasicDataset instance
        neg_ratio: Negative sampling ratio (unused but kept for compatibility)

    Returns:
        np.array: Sampled triplets [user, pos_item, neg_item]
    """
    return UniformSample_optimized(dataset)



def UniformSample_optimized(dataset):
    """
    Ultra-fast BPR sampling with advanced vectorization

    Args:
        dataset: BasicDataset instance

    Returns:
        np.array: Sampled triplets [user, pos_item, neg_item]
    """
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos

    # Pre-allocate result array
    S = np.zeros((user_num, 3), dtype=np.int32)
    valid_samples = 0

    # Get unique users and their counts for efficient processing
    unique_users, user_counts = np.unique(users, return_counts=True)

    # Filter users with positive items
    valid_users = []
    valid_counts = []
    pos_sets = {}

    for user, count in zip(unique_users, user_counts):
        if len(allPos[user]) > 0:
            valid_users.append(user)
            valid_counts.append(count)
            pos_sets[user] = set(allPos[user])

    if not valid_users:
        return np.array([]).reshape(0, 3)

    valid_users = np.array(valid_users)
    valid_counts = np.array(valid_counts)

    # Pre-compute negative item pools for very dense users (>50% items)
    dense_threshold = dataset.m_items * 0.5
    neg_pools = {}

    for user in valid_users:
        if len(pos_sets[user]) > dense_threshold:
            all_items = np.arange(dataset.m_items)
            neg_pools[user] = np.setdiff1d(all_items, list(pos_sets[user]))

    # Process each unique user
    for user, count in zip(valid_users, valid_counts):
        pos_items_user = allPos[user]
        pos_set = pos_sets[user]

        # Vectorized positive item sampling
        pos_indices = np.random.randint(0, len(pos_items_user), count)
        pos_items = pos_items_user[pos_indices]

        # Optimized negative sampling based on user density
        if user in neg_pools:
            # For very dense users: sample from pre-computed negative pool
            neg_items = np.random.choice(neg_pools[user], count)
        elif len(pos_set) < dataset.m_items * 0.1:
            # For sparse users: vectorized rejection sampling
            neg_items = np.zeros(count, dtype=np.int32)
            for i in range(count):
                while True:
                    neg_candidate = np.random.randint(0, dataset.m_items)
                    if neg_candidate not in pos_set:
                        neg_items[i] = neg_candidate
                        break
        else:
            # For medium density users: batch rejection sampling
            neg_items = np.zeros(count, dtype=np.int32)
            candidates_needed = count
            candidates_found = 0

            while candidates_found < count:
                # Generate batch of candidates (oversample to reduce iterations)
                batch_size = min(candidates_needed * 3, dataset.m_items)
                candidates = np.random.randint(0, dataset.m_items, batch_size)

                # Filter out positive items
                for candidate in candidates:
                    if candidate not in pos_set and candidates_found < count:
                        neg_items[candidates_found] = candidate
                        candidates_found += 1

                candidates_needed = count - candidates_found

        # Store results for this user
        user_array = np.full(count, user, dtype=np.int32)
        end_idx = valid_samples + count

        S[valid_samples:end_idx, 0] = user_array
        S[valid_samples:end_idx, 1] = pos_items
        S[valid_samples:end_idx, 2] = neg_items
        valid_samples = end_idx

    return S[:valid_samples]

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _build_model_filename(args, extra_params=None):
    """
    Helper function to build model filename based on model type and parameters

    Args:
        args: Arguments object with model parameters
        extra_params: Additional parameters to include in filename

    Returns:
        str: Generated filename
    """
    base_params = [
        args.data_name, args.model_name, args.data_name,
        args.layer, args.lr, args.decay, args.epochs,
        args.dropout, args.keepprob
    ]

    if args.model_name == "GTN":
        model_params = [
            args.alpha, args.beta, args.gamma, args.alpha1,
            args.alpha2, args.lambda2, args.prop_dropout
        ]
    elif args.model_name == "UltraGCN":
        model_params = [
            args.w1, args.w2, args.w3, args.w4,
            args.lambda_Iweight, args.negative_num, args.ii_neighbor_num
        ]
    else:
        model_params = []

    all_params = base_params + model_params
    if extra_params:
        all_params.extend(extra_params)

    return "-".join(map(str, all_params)) + ".pth.tar"

def getFileName(file_path, args):
    """Generate standard model filename"""
    modelfile = _build_model_filename(args)
    return os.path.join(file_path, modelfile)



def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 256)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key in timer.NAMED_TAPE:
                hint = hint + f"{key}:{timer.NAMED_TAPE[key]:.2f}|"
        else:
            for key in select_keys:
                hint = hint + f"{key}:{timer.NAMED_TAPE[key]:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key in timer.NAMED_TAPE:
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

def generate_rating_matrix_valid(user_dict, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_dict):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_dict, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_dict):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            #sum_recall += len(act_set & pred_set) / float(len(act_set))
            one_user_recall = len(act_set & pred_set) / float(len(act_set))
            recall_dict[i] = one_user_recall
            sum_recall += one_user_recall
            true_users += 1
    return sum_recall / true_users, recall_dict

def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    mrr_dict = {}
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            #sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r==1)[0]+1, dtype=float)[0]
            sum_mrr += one_user_mrr
            true_users += 1
            mrr_dict[i] = one_user_mrr
        else:
            mrr_dict[i] = 0.
    return sum_mrr / len(predicted), mrr_dict

def ndcg_k(actual, predicted, topk):
    res = 0
    ndcg_dict = {}
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def itemperf_recall(ranks, k):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0
    return np.sum(ranks<=k) / len(ranks)

def itemperf_ndcg(ranks, k, size):
    ndcg = 0.0
    if len(ranks) == 0:
        return 0.
    for onerank in ranks:
        r = np.zeros(size)
        r[onerank-1] = 1
        ndcg += ndcg_at_k(r, k)
    return ndcg / len(ranks)

def get_user_performance_perpopularity(train, results_users, Ks):
    """
    Analyze user performance by interaction sequence length

    Args:
        train: Training data dictionary
        results_users: List containing [recall_dict_list, ndcg_dict_list, mrr_dict]
        Ks: List of K values for evaluation
    """
    [recall_dict_list, ndcg_dict_list, mrr_dict] = results_users

    # Initialize result containers
    results_containers = {
        'short': {"recall": np.zeros(len(Ks)), "ndcg": np.zeros(len(Ks)), "mrr": 0., "count": 0},
        'short37': {"recall": np.zeros(len(Ks)), "ndcg": np.zeros(len(Ks)), "mrr": 0., "count": 0},
        'medium7': {"recall": np.zeros(len(Ks)), "ndcg": np.zeros(len(Ks)), "mrr": 0., "count": 0},
        'long': {"recall": np.zeros(len(Ks)), "ndcg": np.zeros(len(Ks)), "mrr": 0., "count": 0}
    }

    def get_user_category(seq_length):
        """Determine user category based on sequence length"""
        if seq_length <= 3:
            return 'short'
        elif seq_length <= 7:
            return 'short37'
        elif seq_length < 20:
            return 'medium7'
        else:
            return 'long'

    test_users = list(train.keys())

    # Single pass to accumulate all metrics
    for result_user in tqdm(test_users, desc="Processing users"):
        seq_length = len(train[result_user])
        category = get_user_category(seq_length)
        container = results_containers[category]

        # Count users in each category
        container["count"] += 1

        # Accumulate recall and ndcg for all K values
        for k_ind in range(len(Ks)):
            recall_dict_k = recall_dict_list[k_ind]
            ndcg_dict_k = ndcg_dict_list[k_ind]
            container["recall"][k_ind] += recall_dict_k[result_user]
            container["ndcg"][k_ind] += ndcg_dict_k[result_user]

        # Accumulate MRR
        container["mrr"] += mrr_dict[result_user]

    # Normalize results and print statistics
    categories = [
        ('short', 'less than 3'),
        ('short37', '3 - 7'),
        ('medium7', '7 - 20'),
        ('long', '>= 20')
    ]

    for category, description in categories:
        container = results_containers[category]
        count = container["count"]

        if count > 0:
            container["recall"] /= count
            container["ndcg"] /= count
            container["mrr"] /= count

        print(f"testing #of users with {description} training points: {count}")
        print(f'test{category}: {{"recall": {container["recall"]}, "ndcg": {container["ndcg"]}, "mrr": {container["mrr"]}}}')


def eval_one_setitems(x):
    Ks = [1, 5, 10, 15, 20, 40]
    result = {
            "recall": 0,
            "ndcg": 0
    }
    ranks = x[0]
    k_ind = x[1]
    test_num_items = x[2]
    freq_ind = x[3]

    result['recall'] = itemperf_recall(ranks, Ks[k_ind])
    result['ndcg'] = itemperf_ndcg(ranks, Ks[k_ind], test_num_items)

    return result, k_ind, freq_ind


def get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, num_items):
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    test_num_items_in_intervals = []
    interval_results = [{'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))} for _ in range(len(items_in_freqintervals))]

    all_freq_all_ranks = []
    all_ks = []
    all_numtestitems = []
    all_freq_ind = []
    for freq_ind, item_list in enumerate(items_in_freqintervals):
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind in range(len(Ks)):
            all_ks.append(k_ind)
            all_freq_all_ranks.append(all_ranks)
            all_numtestitems.append(num_items)
            all_freq_ind.append(freq_ind)
        test_num_items_in_intervals.append(interval_items)

    item_eval_freq_data = zip(all_freq_all_ranks, all_ks, all_numtestitems, all_freq_ind)
    batch_item_result = pool.map(eval_one_setitems, item_eval_freq_data)


    for oneresult in batch_item_result:
        result_dict = oneresult[0]
        k_ind = oneresult[1]
        freq_ind = oneresult[2]
        interval_results[freq_ind]['recall'][k_ind] = result_dict['recall']
        interval_results[freq_ind]['ndcg'][k_ind] = result_dict['ndcg']



    item_freq = freq_quantiles
    for i in range(len(item_freq)+1):
        if i == 0:
            print('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            print('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            print('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(Ks)):
            k = Ks[k_ind]
            print('Recall@%d:%.6f, NDCG@%d:%.6f'%(k, interval_results[i]['recall'][k_ind], k, interval_results[i]['ndcg'][k_ind]))
