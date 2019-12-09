import config as cfg
import story_loader
import data_loader
import csv

mqa = data_loader.DataLoader()


## LOAD THE VALIDATION DATASET
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'plot')
labels = [q[3] for q in qa]
print(labels)
f = open("evaluation_file.txt", "w")
counter = 0
for l in labels:
    f.write("test:" + str(counter) + " " + str(l) + "\n")
    counter = counter + 1
f.close()
