import config 
from models.ssakt import SSAKT
from dataset import get_dataloaders

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score

import os
import numpy as np
import pandas as pd
from datetime import datetime
import wandb

class SAKTModel(pl.LightningModule):
  def __init__(self,model_args,model="ssakt"):
    super().__init__()
    self.model = SSAKT(**model_args)
      
  def forward(self,exercise,category,tag,chapter,test,response,etime,\
    assessmentItemID_answer_mean,user_category_acc,test_elapsed,testId_answer_mean,\
      assessmentItemID_answer_sum,assessmentItemID_test_elapsed_mean,\
        user_testId_acc,user_category_mean_telapsed,user_category_correct_answer,\
          testId_answer_sum,testId_test_elapsed_mean,user_testId_mean_telapsed,\
            user_acc,chapter_answer_mean,user_chapter_acc,user_category_cum_telapsed,\
              tag_answer_mean,user_category_total_answer,tag_answer_sum,\
                chapter_answer_sum,user_correct_answer,last_prob,\
                  user_tag_acc,num_tags_per_test,category_answer_mean):
    return self.model(exercise,category,tag,chapter,test,response,etime,\
      assessmentItemID_answer_mean,user_category_acc,test_elapsed,testId_answer_mean,\
        assessmentItemID_answer_sum,assessmentItemID_test_elapsed_mean,\
          user_testId_acc,user_category_mean_telapsed,user_category_correct_answer,\
            testId_answer_sum,testId_test_elapsed_mean,user_testId_mean_telapsed,\
              user_acc,chapter_answer_mean,user_chapter_acc,user_category_cum_telapsed,\
                tag_answer_mean,user_category_total_answer,tag_answer_sum,\
                  chapter_answer_sum,user_correct_answer,last_prob,\
                    user_tag_acc,num_tags_per_test,category_answer_mean)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters())
  
  def training_step(self,batch,batch_idx):
    inputs,target_ids,target,last_question = batch
    output = self(inputs["input_ids"],inputs["input_cat"],inputs["input_tag"],inputs["input_chap"],inputs["input_test"],target_ids,inputs["input_rtime"],
                  inputs["input_assessmentItemID_answer_mean"],inputs["input_user_category_acc"],inputs["input_test_elapsed"],
                  inputs["input_testId_answer_mean"], inputs["input_assessmentItemID_answer_sum"], inputs["input_assessmentItemID_test_elapsed_mean"],
                  inputs["input_user_testId_acc"], inputs["input_user_category_mean_telapsed"], inputs["input_user_category_correct_answer"],
                  inputs["input_testId_answer_sum"], inputs["input_testId_test_elapsed_mean"], inputs["input_user_testId_mean_telapsed"],
                  inputs["input_user_acc"], inputs['input_chapter_answer_mean'], inputs["input_user_chapter_acc"], inputs['input_user_category_cum_telapsed'],
                  inputs["input_tag_answer_mean"], inputs["input_user_category_total_answer"], inputs["input_tag_answer_sum"],
                  inputs["input_chapter_answer_sum"], inputs["input_user_correct_answer"], inputs["input_last_prob"],
                  inputs["input_user_tag_acc"], inputs["input_num_tags_per_test"], inputs["input_category_answer_mean"])
    target_mask = (target_ids != 0)
    output = torch.masked_select(output.squeeze(),target_mask)
    target = torch.masked_select(target,target_mask)
    loss = nn.BCEWithLogitsLoss()(output.float(),target.float())
    return {"loss":loss,"output":output,"target":target}

  def training_epoch_end(self, training_ouput):
      output = np.concatenate([i["output"].cpu().detach().numpy()
                              for i in training_ouput]).reshape(-1)
      target = np.concatenate([i["target"].cpu().detach().numpy()
                                  for i in training_ouput]).reshape(-1)
      print("train output length", len(output), output.shape)
      accuracy = accuracy_score(target, [1 if x > 0.5 else 0 for x in output])
      auc = roc_auc_score(target, output)
      self.print("train auc: ", auc, " acc: ", accuracy)
      self.log("train_auc", auc)
      self.log("train_acc", accuracy)
  
  def validation_step(self,batch,batch_idx):
    inputs,target_ids,target,last_question = batch
    output = self(inputs["input_ids"],inputs["input_cat"],inputs["input_tag"],inputs["input_chap"],inputs["input_test"],target_ids,inputs["input_rtime"],
                  inputs["input_assessmentItemID_answer_mean"],inputs["input_user_category_acc"],inputs["input_test_elapsed"],
                  inputs["input_testId_answer_mean"], inputs["input_assessmentItemID_answer_sum"], inputs["input_assessmentItemID_test_elapsed_mean"],
                  inputs["input_user_testId_acc"], inputs["input_user_category_mean_telapsed"], inputs["input_user_category_correct_answer"],
                  inputs["input_testId_answer_sum"], inputs["input_testId_test_elapsed_mean"], inputs["input_user_testId_mean_telapsed"],
                  inputs["input_user_acc"], inputs['input_chapter_answer_mean'], inputs["input_user_chapter_acc"], inputs['input_user_category_cum_telapsed'],
                  inputs["input_tag_answer_mean"], inputs["input_user_category_total_answer"], inputs["input_tag_answer_sum"],
                  inputs["input_chapter_answer_sum"], inputs["input_user_correct_answer"], inputs["input_last_prob"],
                  inputs["input_user_tag_acc"], inputs["input_num_tags_per_test"], inputs["input_category_answer_mean"])
    target_mask = (last_question == -1)
    output = torch.masked_select(output.squeeze(),target_mask)
    target = torch.masked_select(target,target_mask)
    loss = nn.BCEWithLogitsLoss()(output.float(),target.float())
    return {"val_loss":loss,"output":output,"target":target}

  def validation_epoch_end(self, validation_ouput):
      output = np.concatenate([i["output"].cpu().detach().numpy()
                            for i in validation_ouput]).reshape(-1)
      target = np.concatenate([i["target"].cpu().detach().numpy()
                               for i in validation_ouput]).reshape(-1)
      print("val output length", len(output), output.shape)
      accuracy = accuracy_score(target, [1 if x > 0.5 else 0 for x in output])
      auc = roc_auc_score(target, output)
      self.print("val auc: ", auc, " acc: ", accuracy)
      self.log("val_auc", auc)
      self.log("val_acc", accuracy)

  def predict_step(self,batch,batch_idx):
      inputs,target_ids,target,last_question = batch
      output = self(inputs["input_ids"],inputs["input_cat"],inputs["input_tag"],inputs["input_chap"],inputs["input_test"],target_ids,inputs["input_rtime"],
                    inputs["input_assessmentItemID_answer_mean"],inputs["input_user_category_acc"],inputs["input_test_elapsed"],
                    inputs["input_testId_answer_mean"], inputs["input_assessmentItemID_answer_sum"], inputs["input_assessmentItemID_test_elapsed_mean"],
                    inputs["input_user_testId_acc"], inputs["input_user_category_mean_telapsed"], inputs["input_user_category_correct_answer"],
                    inputs["input_testId_answer_sum"], inputs["input_testId_test_elapsed_mean"], inputs["input_user_testId_mean_telapsed"],
                    inputs["input_user_acc"], inputs['input_chapter_answer_mean'], inputs["input_user_chapter_acc"], inputs['input_user_category_cum_telapsed'],
                    inputs["input_tag_answer_mean"], inputs["input_user_category_total_answer"], inputs["input_tag_answer_sum"],
                    inputs["input_chapter_answer_sum"], inputs["input_user_correct_answer"], inputs["input_last_prob"],
                    inputs["input_user_tag_acc"], inputs["input_num_tags_per_test"], inputs["input_category_answer_mean"])
      target_mask = (last_question == -1)
      output = torch.masked_select(output.squeeze(),target_mask)
      return output


########### TRAINING AND SAVING MODEL #######

if __name__ == "__main__":

    # args = parse_args()

    wandb.login()
    # wandb.init(project=config.model, config=vars(args), entity="recsys_01")

    train_loader, val_loader, test_loader = get_dataloaders()

    ARGS = {"n_dims":config.EMBED_DIMS ,
                'n_encoder':config.NUM_ENCODER,
                'n_decoder':config.NUM_ENCODER,
                'enc_heads':config.ENC_HEADS,
                'dec_heads':config.ENC_HEADS,
                'total_ex':9455,
                'total_cat':1538,
                'total_tag':913,
                'total_chap':448,
                'total_responses':9455,
                'total_test':199,
                'seq_len':config.MAX_SEQ}

    model_name = config.model
    print("start training ", model_name, " model")
    wandb_logger = WandbLogger(project=model_name, entity="recsys_01")
    datenow = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    checkpoint = ModelCheckpoint(verbose=True,
                                save_top_k=3,
                                monitor="val_auc",
                                dirpath="./checkpoint/"+model_name+"/ckpt/",
                                filename=datenow+"-{epoch:02d}-{val_auc:.5f}",
                                mode="max"
                                )

    sakt_model = SAKTModel(model=model_name,model_args=ARGS)
    trainer = pl.Trainer(gpus=-1,progress_bar_refresh_rate=21,max_epochs=config.n_epochs,callbacks=[checkpoint],logger=wandb_logger) 
    trainer.fit(model = sakt_model, train_dataloaders=train_loader, val_dataloaders=val_loader) 
    trainer.save_checkpoint("./checkpoint/"+model_name+"/saved_model/model_"+model_name+".pt")
    predictions = trainer.predict(sakt_model, test_loader)

    result = []
    for x in predictions:
        result.extend(x.tolist())

    sample_submission = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
    sample_submission["prediction"] = result

    path = "./checkpoint/"+model_name+"/submission/"

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
    
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")

    sample_submission.to_csv(path+datenow+"_"+model_name+".csv", index=False)