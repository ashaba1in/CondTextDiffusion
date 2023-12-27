import torch
from torch.autograd import Function
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLayer
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, upsample,
                 use_do=False, stride=None, kernel_size=None, padding=None, attention=False):
        super().__init__()
        
        conv = nn.Conv1d if not upsample else nn.ConvTranspose1d
        
        if stride is None:
            stride = 2 if downsample or upsample else 1
        if kernel_size is None:
            kernel_size = 3 if not upsample else 2
        if padding is None:
            padding = 1 if not upsample else 0

        self.conv_bn1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5 if use_do else 0),
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        )

        self.conv_bn2 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5 if use_do else 0),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

        self.identity_layer = nn.Identity()
        if in_channels != out_channels or downsample or upsample:
            self.identity_layer = conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        self.relu = nn.ReLU()
        
        self.attention = attention
        if attention:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            config.hidden_size = out_channels
            config.intermediate_size = min(3072, out_channels * 2)
            config.num_attention_heads = 8
            self.bert_layer = BertLayer(config)

    def forward(self, x):
        identity = self.identity_layer(x)
    
        out = self.conv_bn1(x)
        out = self.conv_bn2(out) + identity
        
        if self.attention:
            out = out.permute(0, 2, 1)  # [bs, seq_len, d]
            out = self.bert_layer(out)[0]
            out = out.permute(0, 2, 1)  # [bs, d, seq_len]

        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters, n_block, size_change_gap=2,
                 increasefilter_gap=4, down=True, use_do=False, attention=False):
        super().__init__()
        
        self.n_block = n_block
        self.use_do = use_do
        self.in_channels = in_channels

        self.size_change_gap = size_change_gap
        self.increasefilter_gap = increasefilter_gap

        # first block
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=1, stride=1)

        # self.first_block = BasicBlock(
        #     in_channels=base_filters, 
        #     out_channels=base_filters, 
        #     downsample=down,
        #     upsample=not down,
        #     use_do=self.use_do,
        #     stride=3,
        #     kernel_size=3,
        #     padding=0,
        # )

        # residual blocks
        self.blocks = nn.ModuleList()
        for i_block in range(self.n_block):
            if i_block % self.size_change_gap == 0:
                downsample = down
                upsample = not down
            else:
                downsample = False
                upsample = False

            # increase filters at every self.increasefilter_gap blocks
            in_channels = int(base_filters * 2**(i_block // self.increasefilter_gap))
            if ((i_block + 1) % self.increasefilter_gap == 0) and (i_block != 0):
                out_channels = in_channels * 2
            else:
                out_channels = in_channels
            
            block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                downsample=downsample,
                upsample=upsample,
                use_do=self.use_do,
                attention=attention
            )
            self.blocks.append(block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv1d(out_channels, self.in_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        out = self.conv1(x)
        
        # out = self.first_block(out)
        for block in self.blocks:
            out = block(out)

        out = self.final_bn(out)
        out = self.relu(out)
        out = self.final_conv(out)

        return out



class CNNAutoencoder(nn.Module):
    def __init__(self, encoder_model_name='bert-base-uncased',
                 base_channels=768, vae=False, vqvae=False, K=512, non_util_theshold=50,
                 n_block_encoder=3, size_change_gap_encoder=1, increasefilter_gap_encoder=2,
                 n_block_decoder=3, size_change_gap_decoder=1, increasefilter_gap_decoder=2,
                 attention_decoder=False, head='mlm'
    ):
        super().__init__()
    
        config = AutoConfig.from_pretrained(encoder_model_name)
        self.hid_size = config.hidden_size

        self.cnn_encoder = ResNet1D(
            in_channels=self.hid_size,
            base_filters=base_channels,
            n_block=n_block_encoder,
            size_change_gap=size_change_gap_encoder,
            increasefilter_gap=increasefilter_gap_encoder,
            down=True
        )

        if vqvae:
            # self.codebook = VQEmbedding(K, self.hid_size)
            self.codebook = VectorQuantizer(self.hid_size, K, use_ema=True, decay=0.99, epsilon=1e-5)
            self.K = K
            self.non_utilized_codes_steps = torch.zeros(K)
            self.non_util_theshold = non_util_theshold
        else:
            self.codebook = None

        self.cnn_decoder = ResNet1D(
            in_channels=self.hid_size,
            base_filters=base_channels,
            n_block=n_block_decoder,
            size_change_gap=size_change_gap_decoder,
            increasefilter_gap=increasefilter_gap_decoder,
            down=False,
            attention=attention_decoder,
        )
        
        self.cls = BertDecoder(mode=head)
        # self.cls = BertOnlyMLMHead(self.transformer.config)

        self.vae = vae
        if vae:
            self.relu = nn.ReLU()
            self.mean = nn.Linear(self.hid_size, self.hid_size)
            self.log_var = nn.Linear(self.hid_size, self.hid_size)
        
    def reparametrize(self, mean, log_var):
        noise = torch.randn_like(mean)
        return mean + torch.exp(log_var * 0.5) * noise

    def forward(self, x):
        latent = self.cnn_encoder(x.permute(0, 2, 1)).permute(0, 2, 1)  # [bs, d, seq_len]
        if self.codebook is not None:
            quant_latent_st, dictionary_loss, commitment_loss, encoding_indices = self.codebook(latent)
            
            # restart non-utilized codes
            unique_idxs = torch.unique(encoding_indices.view(-1))
            self.non_utilized_codes_steps += 1
            self.non_utilized_codes_steps[unique_idxs.cpu()] -= 1

            non_utilized_codes = torch.arange(
                len(self.non_utilized_codes_steps)
            )[self.non_utilized_codes_steps > self.non_util_theshold]
            self.non_utilized_codes_steps[non_utilized_codes] = 0
            self.codebook.e_i_ts[:, non_utilized_codes] = torch.FloatTensor(
                self.hid_size, len(non_utilized_codes)
            ).uniform_(-1/self.K, 1/self.K).to(x.device)
            
            # quant_latent_st, quant_latent = self.codebook.straight_through(latent)
            x_rec = self.cnn_decoder(quant_latent_st.permute(0, 2, 1)).permute(0, 2, 1)
            return x_rec, dictionary_loss, commitment_loss, quant_latent_st
        else:
            x_rec = self.cnn_decoder(latent.permute(0, 2, 1)).permute(0, 2, 1)
            return self.cls(x_rec)

    def encode(self, x):
        x = self.cnn_encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.codebook is not None:
            x, _, _, _ = self.codebook(x)

        if not self.vae:
            return x
        else:
            x = self.relu(x)
            mean = self.mean(x)
            log_var = self.log_var(x)

            return mean, log_var

    def decode(self, x):
        if self.codebook is not None:
            x, _, _, _ = self.codebook(x)
        out = self.cnn_decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out

    def decode_to_logits(self, x):
        return self.cls(self.decode(x))


class BertDecoder(nn.Module):
    def __init__(self, mode='mlm', n_layers=3):
        super().__init__()
        self.mode = mode
        config = AutoConfig.from_pretrained('bert-base-uncased')
        if mode == 'transformer':
            config.num_hidden_layers = n_layers
            self.bert = AutoModel.from_config(config).encoder
            # self.bert = AutoModel.from_config(config)
            self.fc = nn.Linear(config.hidden_size, config.vocab_size)
            
            self.net = lambda x: self.fc(self.bert(x).last_hidden_state)

        elif mode == 'mlm':
            self.cls = BertOnlyMLMHead(config)
            self.net = lambda x: self.cls(x)
        else:
            print('Unknown decoder mode', flush=True)
            raise

    def forward(self, x):
        # if self.mode == 'transformer':
        #     position_ids = self.bert.embeddings.position_ids[:, :x.shape[1]]
        #     x += self.bert.embeddings.position_embeddings(position_ids)
        
        return self.net(x)

    def decode_to_logits(self, x):
        return self.forward(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_mode='raw', encoder_path=None, decoder_mode='transformer', decoder_path=None, n_decoder_layers=3):
        super().__init__()
        self.bert_encoder = AutoModel.from_pretrained("bert-base-uncased")
        if encoder_mode == 'cnn':
            n_blocks = 1
            params = dict(
                n_block_encoder=n_blocks, size_change_gap_encoder=1, increasefilter_gap_encoder=100,
                n_block_decoder=3 * n_blocks, size_change_gap_decoder=3, increasefilter_gap_decoder=100,
                base_channels=768
            )

            self.cnn_encoder = CNNAutoencoder(**params)
            if encoder_path is not None:
                self.cnn_encoder.load_state_dict(torch.load(encoder_path))
            self.cnn_encoder = self.cnn_encoder.cnn_encoder

            self.encode = lambda x, attention_mask: self.cnn_encoder(
                self.bert_encoder(x, attention_mask=attention_mask).last_hidden_state.permute(0, 2, 1)
            ).permute(0, 2, 1)
        else:
            self.encode = lambda x, attention_mask: self.bert_encoder(x, attention_mask=attention_mask).last_hidden_state
        
        if decoder_mode != 'cnn':
            self.decoder = BertDecoder(mode=decoder_mode, n_layers=n_decoder_layers)
            if decoder_path is not None:
                self.decoder.load_state_dict(torch.load(decoder_path))
            self.decode = self.decoder
        else:
            n_blocks = 1
            params = dict(
                n_block_encoder=n_blocks, size_change_gap_encoder=1, increasefilter_gap_encoder=100,
                n_block_decoder=3 * n_blocks, size_change_gap_decoder=3, increasefilter_gap_decoder=100,
                base_channels=768
            )

            self.decoder = CNNAutoencoder(**params)
            if decoder_path is not None:
                self.decoder.load_state_dict(torch.load(decoder_path))

            self.decode = lambda x: self.decoder.cls(self.decoder.cnn_decoder(x.permute(0, 2, 1)).permute(0, 2, 1))

    def decode_to_logits(self, x):
        return self.decode(x)
