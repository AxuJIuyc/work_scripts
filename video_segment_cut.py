from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def repick(time):
    s_min = time[0]
    s_sec = time[1]
    e_min = time[2]
    e_sec = time[3]
    return s_min, s_sec, e_min, e_sec

# time_start_sec, time_end_sec:
time = [0, 0, 0, 0]

s_min = time[0]
s_sec = time[1]
e_min = time[2]
e_sec = time[3]

#fulltime = 2018 sec
print(25+27+5+50+19+20+11+50+47+14+120+11+52+7+20+120+120+15+13+32+12+60+40+120+12+57+52+180+4+17+34+60+37+120+50+77+16+20+25+120+10+117, 'sec')

#470-1
time = [0, 20, 0, 45]

s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_1.avi")

#470-2
time = [2, 3, 2, 30]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_2.avi")

#470-3
time = [5, 0, 5, 10]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_3.avi")

#470-4
time = [9, 30, 10, 20]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_4.avi")

#470-5
time = [19, 26, 19, 45]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_5.avi")

#470-6
time = [23, 19, 23, 39]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_6.avi")

#470-7
time = [38, 36, 38, 47]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_7.avi")

#470-8
time = [42, 13, 43, 50]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/truck_1.avi")

#470-9
time = [44, 55, 47, 9]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/washing_chest_1.avi")

#470-10
time = [47, 49, 48, 52]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/washing_chest_2.avi")

#470-11
time = [48, 53, 51, 20]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_130047.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/hose_1.avi")

#471-1
time = [0, 0, 2, 0]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/rope_1.avi")

#471-2
time = [2, 5, 2, 20]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_8.avi")

#471-3
time = [2, 37, 2, 50]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_9.avi")

#471-4
time = [2, 52, 3, 24]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_10.avi")

#471-5
time = [3, 48, 5, 0]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_11.avi")

#471-6
time = [5, 17, 5, 57]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_12.avi")

#471-7
time = [6, 34, 8, 46]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/people_13.avi")

#471-8
time = [16, 3, 17, 52]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/chest_1.avi")

#471-9
time = [31, 38, 34, 42]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/hose_2.avi")

#471-10
time = [34, 43, 36, 34]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/rope_neck_1.avi")

#471-11
time = [40, 3, 41, 5]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/gate_truck_1.avi")

#471-12
time = [41, 23, 44, 0]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/hose_3.avi")

#471-13
time = [49, 10, 51, 17]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300471.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/rope_chest_neck_1.avi")

#472-1
time = [0, 3, 0, 19]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300472.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/rope_2.avi")

#472-0
time = [0, 35, 3, 20]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300472.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/gate_truck_2.avi")

#473-1
time = [10, 3, 12, 10]
s_min, s_sec, e_min, e_sec = repick(time)
ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300473.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/gate_truck_3.avi")

#472-0
# time = [0, 0, 0, 0]
# ffmpeg_extract_subclip("./sourse_video/220316_080000-220316_1300472.avi", (s_min * 60 + s_sec), (e_min * 60 + e_sec), targetname="./cutted_videos/video_1")
