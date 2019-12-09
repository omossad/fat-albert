
import config as cfg
import story_loader
import data_loader
import csv
import numpy as np
from semantic_text_similarity.models import WebBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction
num_sentences = 5
mqa = data_loader.DataLoader()
def stringReplace(input):
    output = input
    if not output:
        output = 'None'
    return output

## LOAD THE VALIDATION DATASET
#movie_list = mqa.get_split_movies(split='val')
#for i in movie_list:
#    mqa.pprint_movie(mqa.movies_map[i])
#print(movie_list)

story, qa = mqa.get_story_qa_data('val', 'split_plot')
#list_of_movies = ['tt0167260','tt0137523','tt1409024','tt0120903','tt0109830','tt0373889']
temp_list = ['tt0373889']
counter = 0
with open('demo/gt.csv', mode='w') as val_file:
    employee_writer = csv.writer(val_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for q in qa:
        movie_idx = q[4]
        if movie_idx in temp_list and counter < 5:
            #movie_story = ', '.join(story[movie_idx])
            question = q[1]
            ans_1 = stringReplace(q[2][0])
            ans_2 = stringReplace(q[2][1])
            ans_3 = stringReplace(q[2][2])
            ans_4 = stringReplace(q[2][3])
            ans_5 = stringReplace(q[2][4])
            index = 0
            movie_align = ''
            sim_mtx = []
            comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
            for sen in story[movie_idx]:
                sim_mtx.append(web_model.predict([(sen, comparison_metric)]).item())
            top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]
            print(q[0],top_sentences)
            for a in top_sentences:
                movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
            combined = " ".join(movie_align.split("\n")) + question
            correct_ans = q[3]
            print(movie_idx)
            print(top_sentences)
            print(movie_align)
            print(question)
            print(ans_1)
            print(ans_2)
            print(ans_3)
            print(ans_4)
            print(ans_5)
            print(correct_ans)
            print('------------')
            counter = counter + 1
            #employee_writer.writerow([movie_idx, top_sentences, movie_align, question, ans_1, ans_2, ans_3, ans_4, ans_5,
            #  correct_ans])


