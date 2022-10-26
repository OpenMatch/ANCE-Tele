import numpy as np
import faiss
from tqdm import tqdm
# from gpu_utils import get_gpu_index
import logging
logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray, use_gpu: bool):
        faiss.omp_set_num_threads(16)
        index = faiss.IndexFlatIP(init_reps.shape[1])
        if use_gpu:
            from gpu_utils import get_gpu_index ## --- SS Modified --- 
            index = get_gpu_index(index)
            logger.info('Gpu Index')
        self.index = index

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size)):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices