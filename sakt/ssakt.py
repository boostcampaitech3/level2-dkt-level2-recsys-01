from multihead_ffn import MultiHeadWithFFN
from utils import pos_encode, get_clones, ut_mask
import config

import torch
from torch import nn

"""
Stacked variant of SAKT(Self-Attentive for Knowledge Tracing)--
The SSAKT model consists of a single attention block that uses exercise embeddings as queries and interaction embeddings as keys/values. 
The authors report a decrease in AUC when the attention block is stacked multiple times. SSAKT resolves this issue 
by applying self-attention on exercises before supplying them as queries. The outputs of the exercise selfattention block and
the exercise-interaction attention block enters the corresponding following blocks as inputs for their attention layers.
"""
class SSAKT(nn.Module):
    def __init__(self,n_encoder,n_decoder,enc_heads,dec_heads,n_dims,total_ex,total_cat,total_tag,total_chap,total_test,total_responses,seq_len):
        super(SSAKT,self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.EncoderBlock_layer = EncoderBlock_layer(enc_heads,n_dims)
        self.ExerciseBlock_layer = ExerciseBlock_layer(dec_heads,n_dims)
        # self.encoder = get_clones(EncoderBlock(enc_heads,n_dims,total_ex,total_cat,total_tag,total_responses,seq_len),n_encoder)
        self.encoder = get_clones(EncoderBlock(n_dims,total_ex,total_cat,total_tag,total_chap,total_test,total_responses,seq_len),n_encoder)
        # self.exercies_block = ExerciseBlock(dec_heads,n_dims,total_ex,seq_len)
        self.exercies_block = ExerciseBlock(n_dims,total_ex,seq_len)
        self.elapsed_time = nn.Linear(1, n_dims)
        self.assessmentItemID_answer_mean = nn.Linear(1, n_dims)
        self.user_category_acc = nn.Linear(1, n_dims)
        self.test_elapsed = nn.Linear(1, n_dims)
        self.testId_answer_mean = nn.Linear(1, n_dims)
        self.assessmentItemID_answer_sum = nn.Linear(1, n_dims)
        self.assessmentItemID_test_elapsed_mean = nn.Linear(1, n_dims)
        self.user_testId_acc = nn.Linear(1, n_dims)
        self.user_category_mean_telapsed = nn.Linear(1, n_dims)
        self.user_category_correct_answer = nn.Linear(1, n_dims)
        self.testId_answer_sum = nn.Linear(1, n_dims)
        self.testId_test_elapsed_mean = nn.Linear(1, n_dims)
        self.user_testId_mean_telapsed = nn.Linear(1, n_dims)
        self.user_acc = nn.Linear(1, n_dims)
        self.chapter_answer_mean = nn.Linear(1, n_dims)
        self.user_chapter_acc = nn.Linear(1, n_dims)
        self.user_category_cum_telapsed = nn.Linear(1, n_dims)
        self.tag_answer_mean = nn.Linear(1, n_dims)
        self.user_category_total_answer = nn.Linear(1, n_dims)
        self.tag_answer_sum = nn.Linear(1, n_dims)
        self.chapter_answer_sum = nn.Linear(1, n_dims)
        self.user_correct_answer = nn.Linear(1, n_dims)
        self.last_prob = nn.Linear(1, n_dims)
        self.user_tag_acc = nn.Linear(1, n_dims)
        self.num_tags_per_test = nn.Linear(1, n_dims)
        self.category_answer_mean = nn.Linear(1, n_dims)
        
        self.fc = nn.Linear(n_dims,1)
    
    def forward(self,in_exercise,in_category,in_tag,in_chap,in_test,in_response,in_etime,\
      in_assessmentItemID_answer_mean,in_user_category_acc,in_test_elapsed,in_testId_answer_mean,\
        in_assessmentItemID_answer_sum,in_assessmentItemID_test_elapsed_mean,\
          in_user_testId_acc,in_user_category_mean_telapsed,in_user_category_correct_answer,\
            in_testId_answer_sum,in_testId_test_elapsed_mean,in_user_testId_mean_telapsed,\
              in_user_acc,in_chapter_answer_mean,in_user_chapter_acc,in_user_category_cum_telapsed,\
                in_tag_answer_mean,in_user_category_total_answer,in_tag_answer_sum,\
                  in_chapter_answer_sum,in_user_correct_answer,in_last_prob,\
                    in_user_tag_acc,in_num_tags_per_test,in_category_answer_mean):
        first_block = True
        exercise = self.exercies_block(input_e=in_exercise)
        exercise = self.ExerciseBlock_layer(exercise)
        for n in range(self.n_encoder):
          if n>=1:
            first_block=False
          enc = self.encoder[n](in_exercise,in_category,in_tag,in_chap,in_test,in_response,first_block=first_block)
          elapsed_time = in_etime.unsqueeze(-1).float()
          ela_time = self.elapsed_time(elapsed_time)
          assessmentItemID_answer_mean = in_assessmentItemID_answer_mean.unsqueeze(-1).float()
          assessmentItemID_answer_mean_linear = self.assessmentItemID_answer_mean(assessmentItemID_answer_mean)
          user_category_acc = in_user_category_acc.unsqueeze(-1).float()
          user_category_acc_linear = self.user_category_acc(user_category_acc)
          test_elapsed = in_test_elapsed.unsqueeze(-1).float()
          test_elapsed_linear = self.test_elapsed(test_elapsed)
          testId_answer_mean = in_testId_answer_mean.unsqueeze(-1).float()
          testId_answer_mean_linear = self.testId_answer_mean(testId_answer_mean)
          assessmentItemID_answer_sum = in_assessmentItemID_answer_sum.unsqueeze(-1).float()
          assessmentItemID_answer_sum_linear = self.assessmentItemID_answer_sum(assessmentItemID_answer_sum)
          assessmentItemID_test_elapsed_mean = in_assessmentItemID_test_elapsed_mean.unsqueeze(-1).float()
          assessmentItemID_test_elapsed_mean_linear = self.assessmentItemID_test_elapsed_mean(assessmentItemID_test_elapsed_mean)
          user_testId_acc = in_user_testId_acc.unsqueeze(-1).float()
          user_testId_acc_linear = self.user_testId_acc(user_testId_acc)
          user_category_mean_telapsed = in_user_category_mean_telapsed.unsqueeze(-1).float()
          user_category_mean_telapsed_linear = self.user_category_mean_telapsed(user_category_mean_telapsed)
          user_category_correct_answer = in_user_category_correct_answer.unsqueeze(-1).float()
          user_category_correct_answer_linear = self.user_category_correct_answer(user_category_correct_answer)
          testId_answer_sum = in_testId_answer_sum.unsqueeze(-1).float()
          testId_answer_sum_linear = self.testId_answer_sum(testId_answer_sum)
          testId_test_elapsed_mean = in_testId_test_elapsed_mean.unsqueeze(-1).float()
          testId_test_elapsed_mean_linear = self.testId_test_elapsed_mean(testId_test_elapsed_mean)
          user_testId_mean_telapsed = in_user_testId_mean_telapsed.unsqueeze(-1).float()
          user_testId_mean_telapsed_linear = self.user_testId_mean_telapsed(user_testId_mean_telapsed)
          user_acc = in_user_acc.unsqueeze(-1).float()
          user_acc_linear = self.user_acc(user_acc)
          chapter_answer_mean = in_chapter_answer_mean.unsqueeze(-1).float()
          chapter_answer_mean_linear = self.chapter_answer_mean(chapter_answer_mean)
          user_chapter_acc = in_user_chapter_acc.unsqueeze(-1).float()
          user_chapter_acc_linear = self.user_chapter_acc(user_chapter_acc)
          user_category_cum_telapsed = in_user_category_cum_telapsed.unsqueeze(-1).float()
          user_category_cum_telapsed_linear = self.user_category_cum_telapsed(user_category_cum_telapsed)
          tag_answer_mean = in_tag_answer_mean.unsqueeze(-1).float()
          tag_answer_mean_linear = self.user_category_cum_telapsed(tag_answer_mean)
          user_category_total_answer = in_user_category_total_answer.unsqueeze(-1).float()
          user_category_total_answer_linear = self.user_category_cum_telapsed(user_category_total_answer)
          tag_answer_sum = in_tag_answer_sum.unsqueeze(-1).float()
          tag_answer_sum_linear = self.tag_answer_sum(tag_answer_sum)
          chapter_answer_sum = in_chapter_answer_sum.unsqueeze(-1).float()
          chapter_answer_sum_linear = self.chapter_answer_sum(chapter_answer_sum)
          user_correct_answer = in_user_correct_answer.unsqueeze(-1).float()
          user_correct_answer_linear = self.user_correct_answer(user_correct_answer)
          last_prob = in_last_prob.unsqueeze(-1).float()
          last_prob_linear = self.last_prob(last_prob)
          user_tag_acc = in_user_tag_acc.unsqueeze(-1).float()
          user_tag_acc_linear = self.user_tag_acc(user_tag_acc)
          num_tags_per_test = in_num_tags_per_test.unsqueeze(-1).float()
          num_tags_per_test_linear = self.num_tags_per_test(num_tags_per_test)
          category_answer_mean = in_category_answer_mean.unsqueeze(-1).float()
          category_answer_mean_linear = self.category_answer_mean(category_answer_mean)

          # enc = enc + ela_time + assessmentItemID_answer_mean_linear + user_category_acc_linear + test_elapsed_linear + testId_answer_mean_linear + assessmentItemID_answer_sum_linear + assessmentItemID_test_elapsed_mean_linear + user_testId_acc_linear + user_category_mean_telapsed_linear
          enc = enc + ela_time + user_category_acc_linear + test_elapsed_linear + user_testId_acc_linear
          enc = self.EncoderBlock_layer(exercise, enc)
          in_exercise = enc
          in_category = enc
          in_tag = enc
          in_chap = enc
          in_test = enc

        return torch.sigmoid(self.fc(enc))



class EncoderBlock(nn.Module):
  def __init__(self,n_dims,total_ex,total_cat,total_tag,total_chap,total_test,total_responses,seq_len):
    super(EncoderBlock,self).__init__()
    self.seq_len = seq_len
    self.exercise_embed = nn.Embedding(total_ex,n_dims)
    self.category_embed = nn.Embedding(total_cat,n_dims)
    self.tag_embed = nn.Embedding(total_tag,n_dims)
    self.chapter_embed = nn.Embedding(total_chap,n_dims)
    self.test_embed = nn.Embedding(total_test,n_dims)
    self.position_embed = nn.Embedding(seq_len,n_dims)
    self.response_embed = nn.Embedding(total_responses,n_dims)
  
  def forward(self,input_e,category,tag,chapter,test,response,first_block=True):
    if first_block:
      _exe = self.exercise_embed(input_e)
      _cat = self.category_embed(category)
      _tag = self.tag_embed(tag)
      _chapter = self.chapter_embed(chapter)
      _test = self.test_embed(test)
      _response = self.response_embed(response)
      position_encoded = pos_encode(self.seq_len-1)
      # args = parse_args()
      if config.device == "cuda":
        position_encoded = position_encoded.cuda()

      _pos = self.position_embed(position_encoded)

      # interaction = _cat + _tag + _chapter + _test + _exe + _response + _pos 
      interaction = _cat + _tag + _exe + _response + _pos
    else:
      interaction = input_e
    return interaction


class EncoderBlock_layer(nn.Module):
  def __init__(self,n_heads,n_dims):
    super(EncoderBlock_layer,self).__init__()
    self.layer_norm = nn.LayerNorm(n_dims)
    self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                      n_dims = n_dims)
  
  def forward(self,exercise,interaction):
    output = self.multihead(q_input=exercise,kv_input=interaction)
    return output


class ExerciseBlock(nn.Module):
    def __init__(self,n_dims,total_exercise,seq_len):
      super(ExerciseBlock,self).__init__()
      self.seq_len = seq_len
      self.exercise_embed = nn.Embedding(total_exercise,n_dims)
      self.position_embed = nn.Embedding(seq_len,n_dims)

    def forward(self,input_e):
        _exe = self.exercise_embed(input_e)
        position_encoded = pos_encode(self.seq_len-1)
        # args = parse_args()
        if config.device == "cuda":
          position_encoded = position_encoded.cuda()
        _pos = self.position_embed(position_encoded)
        exercise = _exe + _pos           
        return exercise

  
class ExerciseBlock_layer(nn.Module):
    def __init__(self,n_heads,n_dims):
      super(ExerciseBlock_layer,self).__init__()
      self.layer_norm = nn.LayerNorm(n_dims)
      self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                              n_dims = n_dims)

    def forward(self,exercise):       
        out_norm = self.layer_norm(exercise)
        output = self.multihead(q_input=exercise,kv_input=exercise)
        output+=out_norm
        return output