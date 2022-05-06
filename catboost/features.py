class Features:
    FEAT = ["user", "assessmentItemID", "testId", "tag", "category", "test", "item", 
        "month", "day", "weekday", "hour", 
        # "elapsed", "test_elapsed",
        # "prev_elapsed", 
        "prev_test_elapsed",
        
        "user_correct_answer", "user_total_answer", "user_acc",
        
        "user_category_correct_answer", "user_category_total_answer", "user_category_acc", 
        # "user_tag_correct_answer", "user_tag_total_answer", "user_tag_acc",
        "user_testId_correct_answer", "user_testId_total_answer", "user_testId_acc", 

        "user_category_cum_telapsed", "user_category_mean_telapsed", 
        # "user_tag_cum_telapsed", "user_tag_mean_telapsed", 
        "user_testId_cum_telapsed", "user_testId_mean_telapsed",

        "testId_answer_mean", "testId_test_elapsed_mean", 
        # "testId_answer_sum",
        
        # "tag_answer_mean", "tag_test_elapsed_mean", 
        # "tag_answer_sum",
        
        "assessmentItemID_answer_mean", "assessmentItemID_test_elapsed_mean", 
        # "assessmentItemID_answer_sum",
        
        "category_answer_mean", "category_test_elapsed_mean", 
        # "category_answer_sum",
        
        # "last_prob"
    ]
    CAT_FEAT = ["user", "assessmentItemID", "testId", "tag", "category", "test", "item", 
                "month", "day", "weekday", "hour"
                ]
