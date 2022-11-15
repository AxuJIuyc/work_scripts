import os
import shutil




#dir with source folders with images  
dir_name = "./cutted_videos/video_frames"
s_dir = os.listdir(dir_name)
s_dir.remove('dataset.csv')
s_dir.remove('people_1-opencv')
s_dir.remove('people_2-opencv')
s_dir.remove('people_3-opencv')
s_dir.remove('people_4-opencv')
s_dir.remove('people_5-opencv')
s_dir.remove('people_6-opencv')
s_dir.remove('people_7-opencv')
s_dir.sort()
# print('Folders:', s_dir)

# перебор файлов в папках
count = 0
for folder in s_dir:
    local_dir = dir_name + '/' + folder

    ls_dir = os.listdir(local_dir)
    
    for img in ls_dir:
        # путь к изображению
        source_path = local_dir + '/' + img
        # путь переноса
        destination_dir = f"./datasets/dataset_imgs/{count}.jpg"
        new_location = shutil.copy(source_path, destination_dir)
        # move(source_path, destination_dir)
        print(f"File '{source_path}' have been replaced into {destination_dir}")
        count += 1
print("Transporting complete!")
    





# # Open statistic_container_file
# print('--> Open statistic file')
# statistic_file = open('statistic.csv', "a+")
# file_writer = csv.writer(statistic_file, delimiter=' ', lineterminator='\r')
# stat_num = 1
# if os.stat('statistic.csv').st_size == 0:
#     file_writer.writerow(["N_exp", "Comments", "avg", "Parametr", "Files"])
#     file_writer.writerow(["-", "-", "-", "-", *s_dir])

# with open('statistic.csv', mode='r') as f:
#     reader = csv.reader(f)
#     nums = [0]
#     for row in reader:
#         print('---row:', row)
#         n = row[0].split()[0]
#         print('---n:', n)
#         if n not in ['N_exp', '-']:
#             nums.append(n)
#     print('---nums:', nums)
#     if nums[-1] not in ['N_exp', '-']:
#         stat_num += int(nums[-1])

#     # file writer
#     avg_realtime = float('%.3f' % (sum(time_list)/len(time_list)))
#     file_writer.writerow([stat_num, "-", avg_realtime, "real time, ms", *time_list])
#     avg_time = float('%.3f' % (sum(total_time)/len(total_time)))
#     file_writer.writerow([stat_num, "-", avg_time, "internal time, ms", *total_time])
#     if EVAL_MEM == True:
#         avg_mem = float('%.3f' % (sum(total_memory)/len(total_memory)))
#         file_writer.writerow([stat_num, "-", avg_mem, "memory, MiB", *total_memory])
#     else:
#         file_writer.writerow([stat_num, "-", 'inactive', "memory, MiB", ])