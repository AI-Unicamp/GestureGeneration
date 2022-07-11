import torch
import torch as th
import torch.nn as nn
import math

############### POSE GENERATIOR #######################
class PoseGenerator(nn.Module):
    def __init__(self, n_frames=61, pose_dim=249, n_mfcc=26, audio_lat_dim=40, time_embed_dim = 40, 
    gru_layers=4, hidden_dim=150, dropout_rate=0.1,):
        super().__init__()
        self.n_frames = n_frames
        self.n_mfcc = n_mfcc
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.audio_lat_dim = audio_lat_dim
        self.time_embed_dim = time_embed_dim
        self.dropout_rate = dropout_rate

        # Audio Treatment Net
        self.audio_encoder = MFCCEncoder(n_frames=self.n_frames, n_mfcc=self.n_mfcc, 
        hidden_dim=self.hidden_dim, output_dim=self.audio_lat_dim, dropout_rate=self.dropout_rate)

        # Timestep embedding net
        
        self.time_embed = nn.Sequential(
            linear(self.time_embed_dim, self.time_embed_dim),
            SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Generation Net
        self.in_size = self.audio_lat_dim + self.pose_dim + self.time_embed_dim
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_dim, num_layers=self.gru_layers, batch_first=True,
                          bidirectional=True, dropout=self.dropout_rate)

        # Project Outputs                  
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim//2, self.pose_dim)
        )


    def forward(self, poses, timesteps, in_audio):
        '''
        :param poses: [BS, N_FRAMES, POSE_DIM]
        :param timesteps: [BS]
        '''
        
        t_embs = self.time_embed(timestep_embedding(timesteps, self.audio_lat_dim)).unsqueeze(1)    # [BS, 1, TIME_EMB_DIM]
        t_embs_broadcast = th.broadcast_to(t_embs,(-1,self.n_frames, self.time_embed_dim))          # [BS, N_FRAMES, TIME_EMB_DIM]
        audio_feat_seq = self.audio_encoder(in_audio)                                               # [BS, N_FRAMES, AUDIO_LAT_DIM]
        in_data = torch.cat((poses, audio_feat_seq,t_embs_broadcast), dim=2)                        # [BS, N_FRAMES, POSE_DIM + AUDIO_LAT_DIM + TIME_EMB_DIM]
        
        decoder_hidden=None
        output, decoder_hidden = self.gru(in_data, decoder_hidden)
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        
        return decoder_outputs

############### AUDIO ENCODER #######################

class MFCCEncoder(nn.Module):
    def __init__(self, n_frames, n_mfcc, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.ff_1 = LinearWithBatchNorm(n_mfcc, hidden_dim, n_frames, dropout_rate)
        self.ff_2 = LinearWithBatchNorm(hidden_dim, hidden_dim, n_frames, dropout_rate)
        self.ff_3 = LinearWithBatchNorm(hidden_dim, hidden_dim, n_frames, dropout_rate)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.ff_1(x)  # batch_size, n_frames, n_mfcc
        x = self.ff_2(x)  # batch_size, n_frames, hidden_dim
        x = self.ff_3(x)  # batch_size, n_frames, hidden_dim

        x = self.out(x)   # batch_size, hidden_dim, out_dim
        return x

class LinearWithBatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim, n_frames, dropout_rate):
        super(LinearWithBatchNorm, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(n_frames)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


############### TIME EMBEDDING ######################
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)