# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.activations import ACT2FN


class Seq2Seq(nn.Module):
    """
        Build Sequence-to-Sequence model with single model as both encoder and decoder.

        Parameters:
        * `model`- the unified model (e.g., T5, BART, or encoder-decoder models)
        * `config`- configuration of the model
        * `beam_size`- beam size for beam search
        * `max_length`- max length of target for beam search
        * `sos_id`- start of symbol ids in target for beam search
        * `eos_id`- end of symbol ids in target for beam search
    """

    def __init__(self, model, config, tokenizer=None,beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer 
        # Register bias for causal attention if needed
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))

        # Check if the model is an encoder-decoder model or encoder-only model
        self.is_encoder_decoder = getattr(config, 'is_encoder_decoder', False)
        self.extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]
        if not self.is_encoder_decoder:
            # For encoder-only models (like BERT, RoBERTa, CodeBERT), we need to add decoder components
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=getattr(config, 'num_attention_heads', 12),
                dim_feedforward=getattr(config, 'intermediate_size', config.hidden_size * 4),
                dropout=getattr(config, 'hidden_dropout_prob', 0.1)
            )
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

            # Output layers
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            # For encoder-decoder models (like T5, BART), use them directly
            self.dense = None
            self.lm_head = None

        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if hasattr(self.config, 'torchscript') and self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if not self.is_encoder_decoder:
            # For encoder-only models, tie weights between embeddings and output layer
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                self._tie_or_clone_weights(self.lm_head, self.model.embeddings.word_embeddings)
            elif hasattr(self.model, 'embed_tokens'):
                self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)
        # For encoder-decoder models, weight tying is handled internally

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        if self.is_encoder_decoder:
            # For encoder-decoder models (T5, BART, etc.)
            if target_ids is not None:
                # Training mode
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=target_ids,
                    decoder_attention_mask=target_mask
                )
                loss = outputs.loss
                # Calculate active tokens for consistent output format
                active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                return loss, loss * active_loss.sum(), active_loss.sum()
            else:
                return self.generate_with_bad_words(source_ids, source_mask)
        else:
            # For encoder-only models (BERT, RoBERTa, CodeBERT, etc.)
            # Encode the source sequence
            encoder_outputs = self.model(source_ids, attention_mask=source_mask)
            encoder_hidden_states = encoder_outputs[0]  # Get last hidden states

            if target_ids is not None:
                # Training mode - use transformer decoder
                # Create target embeddings
                target_embeddings = self.model.embeddings(target_ids)
                target_embeddings = target_embeddings.transpose(0, 1)  # (seq_len, batch, hidden)
                encoder_hidden_states = encoder_hidden_states.transpose(0, 1)  # (seq_len, batch, hidden)

                # Create causal mask for target
                tgt_len = target_ids.size(1)
                tgt_mask = self.bias[:tgt_len, :tgt_len] == 0

                # Decoder forward pass
                decoder_output = self.decoder(
                    target_embeddings,
                    encoder_hidden_states,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=source_mask == 0
                )

                decoder_output = decoder_output.transpose(0, 1)  # (batch, seq_len, hidden)
                hidden_states = torch.tanh(self.dense(decoder_output))
                lm_logits = self.lm_head(hidden_states)

                # Compute loss
                active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])

                return loss, loss * active_loss.sum(), active_loss.sum()
            else:
                # Inference mode - generate sequences
                preds = []
                zero = torch.cuda.LongTensor(1).fill_(0)
                for i in range(source_ids.shape[0]):
                    context = encoder_hidden_states[i:i + 1, :, :].transpose(0, 1)  # (seq_len, 1, hidden)
                    context_mask = source_mask[i:i + 1, :]

                    beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                    input_ids = beam.getCurrentState()
                    context = context.repeat(1, self.beam_size, 1)  # (seq_len, beam_size, hidden)
                    context_mask = context_mask.repeat(self.beam_size, 1)

                    for _ in range(self.max_length):
                        if beam.done():
                            break

                        # Get target embeddings
                        target_embeddings = self.model.embeddings(input_ids).transpose(0, 1)
                        tgt_len = input_ids.size(1)
                        tgt_mask = self.bias[:tgt_len, :tgt_len] == 0

                        # Decoder forward pass
                        decoder_output = self.decoder(
                            target_embeddings,
                            context,
                            tgt_mask=tgt_mask,
                            memory_key_padding_mask=context_mask == 0
                        )

                        decoder_output = decoder_output.transpose(0, 1)  # (batch, seq_len, hidden)
                        hidden_states = torch.tanh(self.dense(decoder_output))
                        logits = self.lm_head(hidden_states)

                        out = self.lsm(logits[:, -1, :]).data
                        beam.advance(out)
                        input_ids = torch.cat([
                            input_ids.index_select(0, beam.getCurrentOrigin()),
                            beam.getCurrentState()
                        ], dim=-1)

                    hyp = beam.getHyp(beam.getFinal())
                    pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                    pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p
                            in pred]
                    preds.append(torch.cat(pred, 0).unsqueeze(0))

                return torch.cat(preds, 0)
    def generate_with_bad_words(self, source_ids, source_mask):
        """使用内置generate方法并过滤占位符"""
        # 准备禁止生成的token
        bad_words_ids = []
        for token in self.extra_id_tokens:
            token_ids = self.tokenizer(token).input_ids
            if token_ids:
                bad_words_ids.append(token_ids)
        
        # 如果没有找到占位符，添加一个虚拟ID防止错误
        if not bad_words_ids:
            bad_words_ids = [[self.model.config.vocab_size + 1000]]
        
        # 生成文本
        outputs = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=self.max_length + 10,  # 增加长度限制
            num_beams=self.beam_size,
            early_stopping=True,
            bad_words_ids=bad_words_ids,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            temperature=0.9,
            num_return_sequences=1,
        )
        return outputs

    def clean_generated_text(self, text, tokenizer):
        """清理生成的文本"""
        # 移除特殊标记
        special_tokens = [
            tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
            "<s>", "</s>", "<pad>", "<unk>"
        ]
        for token in special_tokens:
            text = text.replace(token, "")
        
        # 移除占位符
        text = re.sub(r"<extra_id_\d+>", "", text)
        
        # 清理多余的空格和符号
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[^a-zA-Z0-9]+", "", text)  # 移除开头的非字母数字字符
        text = re.sub(r"[^a-zA-Z0-9\.\?\!]+$", "", text)  # 移除结尾的非字母数字字符
        
        # 处理多余的括号
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        
        return text


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence