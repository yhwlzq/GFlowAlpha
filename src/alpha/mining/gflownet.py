import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from dataclasses import dataclass, field
from alpha.config import Config
from .preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class FeatureMetadataExtractor:
    def __init__(self, preprocessor: DataPreprocessor):
        self.prep = preprocessor
        self.metadata = None

    def build_metadata_matrix(self) -> torch.Tensor:
        logger.info("构建特征元数据矩阵 (IC/ICIR/正交性先验)...")
        X_tr = self.prep.full_X[self.prep.tr_mask]
        y_tr = self.prep.full_y[self.prep.tr_mask]
        tr_dates = self.prep.full_dates[self.prep.tr_mask]

        metadata_list = []
        corr_matrix = np.corrcoef(X_tr.T)
        np.fill_diagonal(corr_matrix, 1.0)

        for i in range(X_tr.shape[1]):
            ic = spearmanr(X_tr[:, i], y_tr)[0]
            if not np.isfinite(ic):
                ic = 0.0

            daily_ic_list = []
            for dt in np.unique(tr_dates):
                day_mask = tr_dates == dt
                n_stocks = day_mask.sum()
                if n_stocks >= 30:
                    x_day = X_tr[day_mask, i]
                    y_day = y_tr[day_mask]
                    if np.ptp(x_day) < 1e-10 or np.ptp(y_day) < 1e-10:
                        continue
                    d_ic = spearmanr(x_day, y_day)[0]
                    if np.isfinite(d_ic):
                        daily_ic_list.append(d_ic)
            if len(daily_ic_list) > 1:
                icir = float(np.mean(daily_ic_list) / (np.std(daily_ic_list, ddof=1) + 1e-8))
            else:
                icir = 0.0

            corr_without_self = np.delete(np.abs(corr_matrix[i, :]), i)
            avg_abs_corr = np.mean(corr_without_self) if len(corr_without_self) > 0 else 0.0
            orthogonality_score = 1.0 - avg_abs_corr
            metadata_list.append([
                ic * 10,
                icir,
                np.tanh(np.mean(np.abs(X_tr[:, i]))),
                0.0,
                orthogonality_score
            ])

        metadata_np = np.array(metadata_list, dtype=np.float32)
        metadata_np = np.nan_to_num(metadata_np, nan=0.0, posinf=1.0, neginf=-1.0)
        mean_vals = np.mean(metadata_np, axis=0)
        std_vals = np.std(metadata_np, axis=0)
        std_vals[std_vals < 1e-8] = 1.0
        metadata_np = (metadata_np - mean_vals) / std_vals

        self.metadata = torch.tensor(metadata_np, dtype=torch.float32, device=Config.DEVICE)
        return self.metadata


@dataclass
class Trajectory:
    actions: List[int]
    features: List[str]
    log_pf: torch.Tensor
    log_pb: torch.Tensor
    length: int


class GFlowNetPolicy_Transformer(nn.Module):
    def __init__(self, num_feats: int, attr_dim: int = 5, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_feats = num_feats
        self.STOP_ACTION = self.num_feats
        self.START_TOKEN = self.num_feats + 1
        self.num_policy_actions = self.num_feats + 1
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(self.num_feats + 2, hidden_dim)
        self.attr_proj = nn.Linear(attr_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.candidate_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.log_Z = nn.Parameter(torch.tensor(0.0))

    def forward(self, action_seq: torch.Tensor, feat_attrs: torch.Tensor, available_mask: torch.Tensor):
        batch_size = action_seq.size(0)
        selected_embs = self.emb(action_seq)
        query = selected_embs.mean(dim=1, keepdim=True)
        kv = self.attr_proj(feat_attrs).unsqueeze(0).expand(batch_size, -1, -1)
        attn_output, _ = self.cross_attn(query=query, key=kv, value=kv)
        context = self.norm1(query + attn_output)
        context = self.norm2(context + self.ffn(context))
        context_expanded = context.expand(-1, self.num_feats, -1)
        candidate_scores = self.candidate_head(context_expanded + kv).squeeze(-1)
        stop_score = self.stop_head(context).squeeze(-1)
        logits = torch.cat([candidate_scores, stop_score], dim=-1)
        logits = logits.masked_fill(~available_mask, -1e9)
        return logits


class TrajectorySampler_Transformer:
    def __init__(self, model, feature_pool, feat_attrs, min_select, max_select, device='cpu'):
        self.model = model
        self.feature_pool = feature_pool
        self.feat_attrs = feat_attrs.to(device)
        self.min_select = min_select
        self.max_select = max_select
        self.device = device
        self.STOP_ACTION = model.STOP_ACTION
        self.START_TOKEN = model.START_TOKEN
        self.num_feats = model.num_feats
        self.num_policy_actions = model.num_policy_actions
        self._feature_penalties = {}

    def set_feature_penalties(self, penalty_dict: Dict[int, float]):
        self._feature_penalties = penalty_dict

    def sample(self, batch_size: int = 1) -> List[Trajectory]:
        return [self._sample_single() for _ in range(batch_size)]

    def _sample_single(self) -> Trajectory:
        selected_indices = []
        available_mask = torch.ones(1, self.num_policy_actions, dtype=torch.bool, device=self.device)
        log_pf_prod = torch.tensor(0.0, device=self.device, requires_grad=True)
        log_pb_prod = torch.tensor(0.0, device=self.device)
        action_seq = torch.tensor([[self.START_TOKEN]], dtype=torch.long, device=self.device)

        for step in range(self.max_select + 1):
            logits = self.model(action_seq, self.feat_attrs, available_mask).squeeze(0)
            #prior_bias = self.feat_attrs[:self.num_feats, 1] * 0.5   #=>ICIR
            prior_bias = self.feat_attrs[:self.num_feats, 0] * 0.5    # IC
            logits[:self.num_feats] = logits[:self.num_feats] + prior_bias

            for feat_idx, penalty in self._feature_penalties.items():
                if feat_idx < self.num_feats:
                    logits[feat_idx] -= penalty

            if len(selected_indices) < self.min_select:
                logits[self.STOP_ACTION] = -1e9
            elif len(selected_indices) >= self.max_select:
                logits[:self.STOP_ACTION] = -1e9

            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8)
            action = torch.multinomial(probs, 1).item()
            log_pf_prod = log_pf_prod + torch.log(probs[action])

            if action == self.STOP_ACTION:
                log_pb_prod = log_pb_prod + torch.log(
                    torch.tensor(1.0 / (len(selected_indices) + 1) + 1e-8, device=self.device))
                break

            selected_indices.append(action)
            available_mask[0, action] = False
            log_pb_prod = log_pb_prod + torch.log(
                torch.tensor(1.0 / (len(selected_indices) + 1) + 1e-8, device=self.device))
            action_seq = torch.cat(
                [action_seq, torch.tensor([[action]], dtype=torch.long, device=self.device)], dim=1)

        return Trajectory(
            actions=selected_indices + [self.STOP_ACTION],
            features=[self.feature_pool[i] for i in selected_indices],
            log_pf=log_pf_prod, log_pb=log_pb_prod, length=len(selected_indices) + 1)


class GFlowNetTrainerInternal:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, log_pf, log_pb, reward: float) -> float:
        self.optimizer.zero_grad()
        reward_tensor = torch.tensor(max(reward, 1e-4), device=log_pf.device, dtype=log_pf.dtype)
        tb_loss = (self.model.log_Z + log_pf - torch.log(reward_tensor) - log_pb.to(log_pf.device)) ** 2
        tb_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return tb_loss.item()
