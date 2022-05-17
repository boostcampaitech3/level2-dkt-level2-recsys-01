import config 

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 


class DKTDataset(Dataset):
  def __init__(self,group,n_skills,max_seq=100,min_seq=10):
    self.samples = group
    self.n_skills = n_skills
    self.max_seq = max_seq
    self.data = []

    for que,ans,res_time,exe_cat,exe_tag,chapter,test,\
      assessmentItemID_answer_mean,user_category_acc,test_elapsed,\
        testId_answer_mean,assessmentItemID_answer_sum,assessmentItemID_test_elapsed_mean,\
          user_testId_acc,user_category_mean_telapsed,user_category_correct_answer,\
            testId_answer_sum,testId_test_elapsed_mean,user_testId_mean_telapsed,\
              user_acc,chapter_answer_mean,user_chapter_acc,user_category_cum_telapsed,\
                tag_answer_mean,user_category_total_answer,tag_answer_sum,\
                  chapter_answer_sum,user_correct_answer,last_prob,\
                    user_tag_acc,num_tags_per_test,category_answer_mean,last_question in self.samples:
        if len(que)>=self.max_seq:
            self.data.extend([(que[l:l+self.max_seq],ans[l:l+self.max_seq],res_time[l:l+self.max_seq],exe_cat[l:l+self.max_seq],
                                exe_tag[l:l+self.max_seq],chapter[l:l+self.max_seq],test[l:l+self.max_seq],
                                assessmentItemID_answer_mean[l:l+self.max_seq],user_category_acc[l:l+self.max_seq],test_elapsed[l:l+self.max_seq],
                                testId_answer_mean[l:l+self.max_seq],assessmentItemID_answer_sum[l:l+self.max_seq],assessmentItemID_test_elapsed_mean[l:l+self.max_seq],
                                user_testId_acc[l:l+self.max_seq],user_category_mean_telapsed[l:l+self.max_seq],user_category_correct_answer[l:l+self.max_seq],
                                testId_answer_sum[l:l+self.max_seq],testId_test_elapsed_mean[l:l+self.max_seq],user_testId_mean_telapsed[l:l+self.max_seq],
                                user_acc[l:l+self.max_seq],chapter_answer_mean[l:l+self.max_seq],user_chapter_acc[l:l+self.max_seq],user_category_cum_telapsed[l:l+self.max_seq],
                                tag_answer_mean[l:l+self.max_seq],user_category_total_answer[l:l+self.max_seq],tag_answer_sum[l:l+self.max_seq],
                                chapter_answer_sum[l:l+self.max_seq],user_correct_answer[l:l+self.max_seq],last_prob[l:l+self.max_seq],
                                user_tag_acc[l:l+self.max_seq],num_tags_per_test[l:l+self.max_seq],category_answer_mean[l:l+self.max_seq],last_question[l:l+self.max_seq]
                                )\
            for l in range(len(que)) if l%self.max_seq==0])
        elif len(que)<self.max_seq and len(que)>min_seq:
            self.data.append((que,ans,res_time,exe_cat,exe_tag,chapter,test,assessmentItemID_answer_mean,user_category_acc,test_elapsed,
                              testId_answer_mean,assessmentItemID_answer_sum,assessmentItemID_test_elapsed_mean,
                              user_testId_acc,user_category_mean_telapsed,user_category_correct_answer,
                              testId_answer_sum,testId_test_elapsed_mean,user_testId_mean_telapsed,
                              user_acc,chapter_answer_mean,user_chapter_acc,user_category_cum_telapsed,
                              tag_answer_mean,user_category_total_answer,tag_answer_sum,
                              chapter_answer_sum,user_correct_answer,last_prob,
                              user_tag_acc,num_tags_per_test,category_answer_mean,last_question))
        else :
            continue
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    content_ids,answered_correctly,response_time,exe_category,exe_knowledgetag,exe_chapter,exe_test,\
      exe_assessmentItemID_answer_mean,exe_user_category_acc,exe_test_elapsed,\
        exe_testId_answer_mean,exe_assessmentItemID_answer_sum,exe_assessmentItemID_test_elapsed_mean,\
          exe_user_testId_acc,exe_user_category_mean_telapsed,exe_user_category_correct_answer,\
            exe_testId_answer_sum,exe_testId_test_elapsed_mean,exe_user_testId_mean_telapsed,\
              exe_user_acc,exe_chapter_answer_mean,exe_user_chapter_acc,exe_user_category_cum_telapsed,\
                exe_tag_answer_mean,exe_user_category_total_answer,exe_tag_answer_sum,\
                  exe_chapter_answer_sum,exe_user_correct_answer,exe_last_prob,\
                    exe_user_tag_acc,exe_num_tags_per_test,exe_category_answer_mean,exe_last_question = self.data[idx]
    seq_len = len(content_ids)

    q_ids = np.zeros(self.max_seq,dtype=int)
    ans = np.zeros(self.max_seq,dtype=int)
    last_question = np.zeros(self.max_seq,dtype=int)
    r_time = np.zeros(self.max_seq,dtype=int)
    exe_cat = np.zeros(self.max_seq,dtype=int)
    exe_tag = np.zeros(self.max_seq,dtype=int)
    exe_chap = np.zeros(self.max_seq,dtype=int)
    exe_exam = np.zeros(self.max_seq,dtype=int)
    assessmentItemID_answer_mean = np.zeros(self.max_seq,dtype=int)
    user_category_acc = np.zeros(self.max_seq,dtype=int)
    test_elapsed = np.zeros(self.max_seq,dtype=int)
    testId_answer_mean = np.zeros(self.max_seq,dtype=int)
    assessmentItemID_answer_sum = np.zeros(self.max_seq,dtype=int)
    assessmentItemID_test_elapsed_mean = np.zeros(self.max_seq,dtype=int)
    user_testId_acc = np.zeros(self.max_seq,dtype=int)
    user_category_mean_telapsed = np.zeros(self.max_seq,dtype=int)
    user_category_correct_answer = np.zeros(self.max_seq,dtype=int)
    testId_answer_sum = np.zeros(self.max_seq,dtype=int)
    testId_test_elapsed_mean = np.zeros(self.max_seq,dtype=int)
    user_testId_mean_telapsed = np.zeros(self.max_seq,dtype=int)
    user_acc = np.zeros(self.max_seq,dtype=int)
    chapter_answer_mean = np.zeros(self.max_seq,dtype=int)
    user_chapter_acc = np.zeros(self.max_seq,dtype=int)
    user_category_cum_telapsed = np.zeros(self.max_seq,dtype=int)
    tag_answer_mean = np.zeros(self.max_seq,dtype=int)
    user_category_total_answer = np.zeros(self.max_seq,dtype=int)
    tag_answer_sum = np.zeros(self.max_seq,dtype=int)
    chapter_answer_sum = np.zeros(self.max_seq,dtype=int)
    user_correct_answer = np.zeros(self.max_seq,dtype=int)
    last_prob = np.zeros(self.max_seq,dtype=int)
    user_tag_acc = np.zeros(self.max_seq,dtype=int)
    num_tags_per_test = np.zeros(self.max_seq,dtype=int)
    category_answer_mean = np.zeros(self.max_seq,dtype=int)

    if seq_len>=self.max_seq:
      q_ids[:] = content_ids[-self.max_seq:]
      ans[:] = answered_correctly[-self.max_seq:]
      last_question[:] = exe_last_question[-self.max_seq:]
      r_time[:] = response_time[-self.max_seq:]
      exe_cat[:] = exe_category[-self.max_seq:]
      exe_tag[:] = exe_knowledgetag[-self.max_seq:]
      exe_chap[:] = exe_chapter[-self.max_seq:]
      exe_exam[:] = exe_test[-self.max_seq:]
      assessmentItemID_answer_mean[:] = exe_assessmentItemID_answer_mean[-self.max_seq:]
      user_category_acc[:] = exe_user_category_acc[-self.max_seq:]
      test_elapsed[:] = exe_test_elapsed[-self.max_seq:]
      testId_answer_mean[:] = exe_testId_answer_mean[-self.max_seq:]
      assessmentItemID_answer_sum[:] = exe_assessmentItemID_answer_sum[-self.max_seq:]
      assessmentItemID_test_elapsed_mean[:] = exe_assessmentItemID_test_elapsed_mean[-self.max_seq:]
      user_testId_acc[:] = exe_user_testId_acc[-self.max_seq:]
      user_category_mean_telapsed[:] = exe_user_category_mean_telapsed[-self.max_seq:]
      user_category_correct_answer[:] = exe_user_category_correct_answer[-self.max_seq:]
      testId_answer_sum[:] = exe_testId_answer_sum[-self.max_seq:]
      testId_test_elapsed_mean[:] = exe_testId_test_elapsed_mean[-self.max_seq:]
      user_testId_mean_telapsed[:] = exe_user_testId_mean_telapsed[-self.max_seq:]
      user_acc[:] = exe_user_acc[-self.max_seq:]
      chapter_answer_mean[:] = exe_chapter_answer_mean[-self.max_seq:]
      user_chapter_acc[:] = exe_user_chapter_acc[-self.max_seq:]
      user_category_cum_telapsed[:] = exe_user_category_cum_telapsed[-self.max_seq:]
      tag_answer_mean[:] = exe_tag_answer_mean[-self.max_seq:]
      user_category_total_answer[:] = exe_user_category_total_answer[-self.max_seq:]
      tag_answer_sum[:] = exe_tag_answer_sum[-self.max_seq:]
      chapter_answer_sum[:] = exe_chapter_answer_sum[-self.max_seq:]
      user_correct_answer[:] = exe_user_correct_answer[-self.max_seq:]
      last_prob[:] = exe_last_prob[-self.max_seq:]
      user_tag_acc[:] = exe_user_tag_acc[-self.max_seq:]
      num_tags_per_test[:] = exe_num_tags_per_test[-self.max_seq:]
      category_answer_mean[:] = exe_category_answer_mean[-self.max_seq:]

    else:
      q_ids[-seq_len:] = content_ids
      ans[-seq_len:] = answered_correctly
      last_question[-seq_len:] = exe_last_question
      r_time[-seq_len:] = response_time
      exe_cat[-seq_len:] = exe_category
      exe_tag[-seq_len:] = exe_knowledgetag
      exe_chap[-seq_len:] = exe_chapter
      exe_exam[-seq_len:] = exe_test
      assessmentItemID_answer_mean[-seq_len:] = exe_assessmentItemID_answer_mean
      user_category_acc[-seq_len:] = exe_user_category_acc
      test_elapsed[-seq_len:] = exe_test_elapsed
      testId_answer_mean[-seq_len:] = exe_testId_answer_mean
      assessmentItemID_answer_sum[-seq_len:] = exe_assessmentItemID_answer_sum
      assessmentItemID_test_elapsed_mean[-seq_len:] = exe_assessmentItemID_test_elapsed_mean
      user_testId_acc[-seq_len:] = exe_user_testId_acc
      user_category_mean_telapsed[-seq_len:] = exe_user_category_mean_telapsed
      user_category_correct_answer[-seq_len:] = exe_user_category_correct_answer
      testId_answer_sum[-seq_len:] = exe_testId_answer_sum
      testId_test_elapsed_mean[-seq_len:] = exe_testId_test_elapsed_mean
      user_testId_mean_telapsed[-seq_len:] = exe_user_testId_mean_telapsed
      user_acc[-seq_len:] = exe_user_acc
      chapter_answer_mean[-seq_len:] = exe_chapter_answer_mean
      user_chapter_acc[-seq_len:] = exe_user_chapter_acc
      user_category_cum_telapsed[-seq_len:] = exe_user_category_cum_telapsed
      tag_answer_mean[-seq_len:] = exe_tag_answer_mean
      user_category_total_answer[-seq_len:] = exe_user_category_total_answer
      tag_answer_sum[-seq_len:] = exe_tag_answer_sum
      chapter_answer_sum[-seq_len:] = exe_chapter_answer_sum
      user_correct_answer[-seq_len:] = exe_user_correct_answer
      last_prob[-seq_len:] = exe_last_prob
      user_tag_acc[-seq_len:] = exe_user_tag_acc
      num_tags_per_test[-seq_len:] = exe_num_tags_per_test
      category_answer_mean[-seq_len:] = exe_category_answer_mean
    
    target_qids = q_ids[1:]
    label = ans[1:] 
    last_label = last_question[1:] 

    input_ids = np.zeros(self.max_seq-1,dtype=int)
    input_ids = q_ids[:-1].copy()

    input_rtime = np.zeros(self.max_seq-1,dtype=int)
    input_rtime = r_time[:-1].copy()

    input_cat = np.zeros(self.max_seq-1,dtype=int)
    input_cat = exe_cat[:-1].copy()

    input_tag = np.zeros(self.max_seq-1,dtype=int)
    input_tag = exe_tag[:-1].copy()

    input_chap = np.zeros(self.max_seq-1,dtype=int)
    input_chap = exe_chap[:-1].copy()

    input_test = np.zeros(self.max_seq-1,dtype=int)
    input_test = exe_exam[:-1].copy()

    input_assessmentItemID_answer_mean = np.zeros(self.max_seq-1,dtype=int)
    input_assessmentItemID_answer_mean = assessmentItemID_answer_mean[:-1].copy()

    input_user_category_acc = np.zeros(self.max_seq-1,dtype=int)
    input_user_category_acc = user_category_acc[:-1].copy()

    input_test_elapsed = np.zeros(self.max_seq-1,dtype=int)
    input_test_elapsed = test_elapsed[:-1].copy()

    input_testId_answer_mean = np.zeros(self.max_seq-1,dtype=int)
    input_testId_answer_mean = testId_answer_mean[:-1].copy()

    input_assessmentItemID_answer_sum = np.zeros(self.max_seq-1,dtype=int)
    input_assessmentItemID_answer_sum = assessmentItemID_answer_sum[:-1].copy()

    input_assessmentItemID_test_elapsed_mean = np.zeros(self.max_seq-1,dtype=int)
    input_assessmentItemID_test_elapsed_mean = assessmentItemID_test_elapsed_mean[:-1].copy()

    input_user_testId_acc = np.zeros(self.max_seq-1,dtype=int)
    input_user_testId_acc = user_testId_acc[:-1].copy()

    input_user_category_mean_telapsed = np.zeros(self.max_seq-1,dtype=int)
    input_user_category_mean_telapsed = user_category_mean_telapsed[:-1].copy()

    input_user_category_correct_answer = np.zeros(self.max_seq-1,dtype=int)
    input_user_category_correct_answer = user_category_correct_answer[:-1].copy()

    input_testId_answer_sum = np.zeros(self.max_seq-1,dtype=int)
    input_testId_answer_sum = testId_answer_sum[:-1].copy()

    input_testId_test_elapsed_mean = np.zeros(self.max_seq-1,dtype=int)
    input_testId_test_elapsed_mean = testId_test_elapsed_mean[:-1].copy()

    input_user_testId_mean_telapsed = np.zeros(self.max_seq-1,dtype=int)
    input_user_testId_mean_telapsed = user_testId_mean_telapsed[:-1].copy()

    input_user_acc = np.zeros(self.max_seq-1,dtype=int)
    input_user_acc = user_acc[:-1].copy()

    input_chapter_answer_mean = np.zeros(self.max_seq-1,dtype=int)
    input_chapter_answer_mean = chapter_answer_mean[:-1].copy()

    input_user_chapter_acc = np.zeros(self.max_seq-1,dtype=int)
    input_user_chapter_acc = user_chapter_acc[:-1].copy()

    input_user_category_cum_telapsed = np.zeros(self.max_seq-1,dtype=int)
    input_user_category_cum_telapsed = user_category_cum_telapsed[:-1].copy()

    input_tag_answer_mean = np.zeros(self.max_seq-1,dtype=int)
    input_tag_answer_mean = tag_answer_mean[:-1].copy()

    input_user_category_total_answer = np.zeros(self.max_seq-1,dtype=int)
    input_user_category_total_answer = user_category_total_answer[:-1].copy()

    input_tag_answer_sum = np.zeros(self.max_seq-1,dtype=int)
    input_tag_answer_sum = tag_answer_sum[:-1].copy()

    input_chapter_answer_sum = np.zeros(self.max_seq-1,dtype=int)
    input_chapter_answer_sum = chapter_answer_sum[:-1].copy()

    input_user_correct_answer = np.zeros(self.max_seq-1,dtype=int)
    input_user_correct_answer = user_correct_answer[:-1].copy()

    input_last_prob = np.zeros(self.max_seq-1,dtype=int)
    input_last_prob = last_prob[:-1].copy()

    input_user_tag_acc = np.zeros(self.max_seq-1,dtype=int)
    input_user_tag_acc = user_tag_acc[:-1].copy()

    input_num_tags_per_test = np.zeros(self.max_seq-1,dtype=int)
    input_num_tags_per_test = num_tags_per_test[:-1].copy()

    input_category_answer_mean = np.zeros(self.max_seq-1,dtype=int)
    input_category_answer_mean = category_answer_mean[:-1].copy()

    input = {"input_ids":input_ids,"input_rtime":input_rtime.astype(np.int),"input_cat":input_cat,"input_tag":input_tag,"input_chap":input_chap,"input_test":input_test,
              "input_assessmentItemID_answer_mean":input_assessmentItemID_answer_mean,"input_user_category_acc":input_user_category_acc,"input_test_elapsed":input_test_elapsed,
              "input_testId_answer_mean":input_testId_answer_mean, "input_assessmentItemID_answer_sum":input_assessmentItemID_answer_sum, "input_assessmentItemID_test_elapsed_mean":input_assessmentItemID_test_elapsed_mean,
              "input_user_testId_acc":input_user_testId_acc, "input_user_category_mean_telapsed":input_user_category_mean_telapsed, "input_user_category_correct_answer":input_user_category_correct_answer,
              "input_testId_answer_sum":input_testId_answer_sum, "input_testId_test_elapsed_mean":input_testId_test_elapsed_mean, "input_user_testId_mean_telapsed":input_user_testId_mean_telapsed,
              "input_user_acc":input_user_acc, "input_chapter_answer_mean":input_chapter_answer_mean, "input_user_chapter_acc":input_user_chapter_acc, "input_user_category_cum_telapsed":input_user_category_cum_telapsed,
              "input_tag_answer_mean":input_tag_answer_mean, "input_user_category_total_answer":input_user_category_total_answer, "input_tag_answer_sum":input_tag_answer_sum,
              "input_chapter_answer_sum":input_chapter_answer_sum, "input_user_correct_answer":input_user_correct_answer, "input_last_prob":input_last_prob,
              "input_user_tag_acc":input_user_tag_acc, "input_num_tags_per_test":input_num_tags_per_test, "input_category_answer_mean":input_category_answer_mean}

    return input,target_qids,label,last_label



def get_dataloaders():              
    print("loading csv.....")
    # args = parse_args()
    train_df = pd.read_csv(config.TRAIN_FILE, usecols=['Timestamp', 'user', 'assessmentItemID', 'answer', 'prev_test_elapsed', 'testId', 'tag', 'label',
                                                        'assessmentItemID_answer_mean', 'user_category_acc', 'chapter', 'test_elapsed', 
                                                        'testId_answer_mean', 'assessmentItemID_answer_sum', 'assessmentItemID_test_elapsed_mean',
                                                        'user_testId_acc', 'user_category_mean_telapsed', 'test', 'user_category_correct_answer',
                                                        'testId_answer_sum', 'testId_test_elapsed_mean', 'user_testId_mean_telapsed',
                                                        'user_acc', 'chapter_answer_mean', 'user_chapter_acc', 'user_category_cum_telapsed',
                                                        'tag_answer_mean', 'user_category_total_answer', 'tag_answer_sum',
                                                        'chapter_answer_sum', 'user_correct_answer', 'last_prob',
                                                        'user_tag_acc', 'num_tags_per_test', 'category_answer_mean'])
    print("shape of dataframe :",train_df.shape) 

    train_df.rename(columns = {'Timestamp':'timestamp', 'user':'user_id', 'assessmentItemID':'content_id',
                                'answer':'answered_correctly', 'prev_test_elapsed':'prior_question_elapsed_time',
                                'testId':'task_container_id', 'tag':'tag_id'}, inplace = True)
    train_df["chapter"] = train_df["chapter"].astype("object")
    train_df["test"] = train_df["test"].astype("object")
    le_user_id = LabelEncoder()
    train_df['user_id'] = le_user_id.fit_transform(train_df['user_id']) + 1
    le_content_id = LabelEncoder()
    train_df['content_id'] = le_content_id.fit_transform(train_df['content_id']) + 1
    le_task_container_id = LabelEncoder()
    train_df['task_container_id'] = le_task_container_id.fit_transform(train_df['task_container_id']) + 1
    le_tag_id = LabelEncoder()
    train_df['tag_id'] = le_tag_id.fit_transform(train_df['tag_id']) + 1
    le_chapter = LabelEncoder()
    train_df['chapter'] = le_chapter.fit_transform(train_df['chapter']) + 1
    le_test = LabelEncoder()
    train_df['test'] = le_test.fit_transform(train_df['test']) + 1
    train_df.prior_question_elapsed_time.fillna(0, inplace=True)
    train_df.prior_question_elapsed_time /= 1000
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)
    
    train_df = train_df.sort_values(["user_id", "timestamp"],ascending=True).reset_index(drop=True)
    skills = train_df.content_id.unique()
    n_skills = len(skills)
    n_cats = len(train_df.task_container_id.unique())+100
    n_tags = len(train_df.tag_id.unique())+100
    print("no. of skills :",n_skills)
    print("no. of categories: ", n_cats)
    print("no. of tags: ", n_tags)
    print("shape after exlusion:",train_df.shape)

    test_df = train_df[train_df["label"] == "test"]
    test_df["last_question"] = test_df["answered_correctly"]
    # test_df = test_df[test_df["answered_correctly"] == -1]
    # test_df.answered_correctly = 0
    print("test_df size", test_df.shape)
    train_df = train_df[train_df["label"] == "train"]
    train_df["last_question"] = [-1 if is_bool else 1 for is_bool in train_df.user_id != train_df.user_id.shift(-1)]

    #grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id","tag_id",
                      "assessmentItemID_answer_mean", "user_category_acc", "chapter", "test_elapsed", 
                      "testId_answer_mean", "assessmentItemID_answer_sum", "assessmentItemID_test_elapsed_mean",
                      "user_testId_acc", "user_category_mean_telapsed", "test", "user_category_correct_answer",
                      "testId_answer_sum", "testId_test_elapsed_mean", "user_testId_mean_telapsed",
                      "user_acc", "chapter_answer_mean", "user_chapter_acc", "user_category_cum_telapsed",
                      "tag_answer_mean", "user_category_total_answer", "tag_answer_sum",
                      "chapter_answer_sum", "user_correct_answer", "last_prob",
                      "user_tag_acc", "num_tags_per_test", "category_answer_mean", "last_question"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,r.answered_correctly.values,r.prior_question_elapsed_time.values,r.task_container_id.values,
                                      r.tag_id.values, r.chapter.values, r.test.values,
                                      r.assessmentItemID_answer_mean.values, r.user_category_acc.values, r.test_elapsed.values,
                                      r.testId_answer_mean.values, r.assessmentItemID_answer_sum.values, r.assessmentItemID_test_elapsed_mean.values,
                                      r.user_testId_acc.values, r.user_category_mean_telapsed.values, r.user_category_correct_answer.values,
                                      r.testId_answer_sum.values, r.testId_test_elapsed_mean.values, r.user_testId_mean_telapsed.values,
                                      r.user_acc.values, r.chapter_answer_mean.values, r.user_chapter_acc.values, r.user_category_cum_telapsed.values,
                                      r.tag_answer_mean.values, r.user_category_total_answer.values, r.tag_answer_sum.values,
                                      r.chapter_answer_sum.values, r.user_correct_answer.values, r.last_prob.values,
                                      r.user_tag_acc.values, r.num_tags_per_test.values, r.category_answer_mean.values, r.last_question.values))

    group_test = test_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id","tag_id",
                      "assessmentItemID_answer_mean", "user_category_acc", "chapter", "test_elapsed", 
                      "testId_answer_mean", "assessmentItemID_answer_sum", "assessmentItemID_test_elapsed_mean",
                      "user_testId_acc", "user_category_mean_telapsed", "test", "user_category_correct_answer",
                      "testId_answer_sum", "testId_test_elapsed_mean", "user_testId_mean_telapsed",
                      "user_acc", "chapter_answer_mean", "user_chapter_acc", "user_category_cum_telapsed",
                      "tag_answer_mean", "user_category_total_answer", "tag_answer_sum",
                      "chapter_answer_sum", "user_correct_answer", "last_prob",
                      "user_tag_acc", "num_tags_per_test", "category_answer_mean", "last_question"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,r.answered_correctly.values,r.prior_question_elapsed_time.values,r.task_container_id.values,
                                      r.tag_id.values, r.chapter.values, r.test.values,
                                      r.assessmentItemID_answer_mean.values, r.user_category_acc.values, r.test_elapsed.values,
                                      r.testId_answer_mean.values, r.assessmentItemID_answer_sum.values, r.assessmentItemID_test_elapsed_mean.values,
                                      r.user_testId_acc.values, r.user_category_mean_telapsed.values, r.user_category_correct_answer.values,
                                      r.testId_answer_sum.values, r.testId_test_elapsed_mean.values, r.user_testId_mean_telapsed.values,
                                      r.user_acc.values, r.chapter_answer_mean.values, r.user_chapter_acc.values, r.user_category_cum_telapsed.values,
                                      r.tag_answer_mean.values, r.user_category_total_answer.values, r.tag_answer_sum.values,
                                      r.chapter_answer_sum.values, r.user_correct_answer.values, r.last_prob.values,
                                      r.user_tag_acc.values, r.num_tags_per_test.values, r.category_answer_mean.values, r.last_question.values))
    del train_df, test_df
    gc.collect()

    print("splitting")
    train,val = train_test_split(group,test_size=0.2,random_state=42) 
    print("train size: ",train.shape,"validation size: ",val.shape,"test size: ",group_test.shape)
    print(val.index[:10])
    train_dataset = DKTDataset(train.values,n_skills=n_skills,max_seq = config.MAX_SEQ,min_seq=0)
    val_dataset = DKTDataset(val.values,n_skills=n_skills,max_seq = config.MAX_SEQ,min_seq=0)
    test_dataset = DKTDataset(group_test.values,n_skills=n_skills,max_seq=config.MAX_SEQ,min_seq=0)
    train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    test_loader = DataLoader(test_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    del train_dataset,val_dataset,test_dataset
    gc.collect()
    return train_loader, val_loader, test_loader