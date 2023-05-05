import os
import math
import csv
import cv2
import numpy as np


"""
'height': 24-1,
'width': 13-24,
'top_line_thickness': 2-1,
'bottom_line_thickness': 24-23,
'gap_1': 21-22,
'gap_2': 17-18,
'tail_height': 13-11,
'stiffener_deflection': 29-5 
"""


class Calculations:
    def __init__(self, points, image_size=(3510, 2550), res=300):
        self.default_params = {
            'height': [17, 0.4, 'Высота'], # [default, deviation] millimeters
            'width': [78, 0.6, 'Ширина'],
            'top_line_thickness': [0.7, 0.3, 'Толщина верхней кромки'],
            'bottom_line_thickness': [0.9, 0.3, 'Толщина нижней кромки'],
            'gap_1': [4.3, 0.3, 'Ширина зазора 1'],
            'gap_2': [4.3, 0.3, 'Ширина зазора 2'],
            'tail_height': [15.4, 0.2, 'Высота хвоста'],
            'stiffener_deflection': [0, 5, 'Отклонение ребра'], # [default, deviation] grad
        }
        self.points = self.dict_points(points)
        self.image_size = image_size
        self.resolution = res # points per inch
        self.res_mm = self.resolution / 25.4 # points per mm
        
    def dict_points(self, points):
        dct = {}
        for i in range(len(points)):
            dct.update({f'{i+1}':(int(points[i][0]), int(points[i][1]))})
        return dct
    
    def culc_hw(self):
        height = abs(self.points['24'][1] - self.points['1'][1])
        width = abs(self.points['13'][0] - self.points['24'][0])
        self.height = self.p2mm(height)
        self.width = self.p2mm(width)
    
    def culc_line_thickness(self):
        top_line_thickness = abs(self.points['2'][1] - self.points['1'][1])
        bottom_line_thickness = abs(self.points['24'][1] - self.points['23'][1])
        self.top_line_thickness = self.p2mm(top_line_thickness)
        self.bottom_line_thickness = self.p2mm(bottom_line_thickness)
    
    def culc_gaps(self):
        gap_1 = abs(self.points['21'][0] - self.points['22'][0])
        gap_2 = abs(self.points['17'][0] - self.points['18'][0])
        self.gap_1 = self.p2mm(gap_1)
        self.gap_2 = self.p2mm(gap_2)
    
    def culc_tail(self):
        tail_height = abs(self.points['13'][1] - self.points['11'][1])
        self.tail_height = self.p2mm(tail_height)
        
    def culc_stiffener(self):
        stiffener_1_thickness = abs(self.points['26'][0] - self.points['3'][0])
        stiffener_1_length = abs(self.points['26'][1] - self.points['3'][1])
        stiffener_2_thickness = abs(self.points['28'][0] - self.points['4'][0])
        stiffener_2_length = abs(self.points['28'][1] - self.points['4'][1])
        stiffener_3_thickness = abs(self.points['29'][0] - self.points['5'][0])
        stiffener_3_length = abs(self.points['29'][1] - self.points['5'][1])
        stiffener_4_thickness = abs(self.points['31'][0] - self.points['6'][0])
        stiffener_4_length = abs(self.points['31'][1] - self.points['6'][1])
        
        avg_length = (stiffener_1_length + stiffener_2_length + 
                      stiffener_3_length + stiffener_4_length)/4
        avg_thickness = (stiffener_1_thickness + stiffener_2_thickness + 
                         stiffener_3_thickness + stiffener_4_thickness)/4
        deflection = self.points['5'][0] + avg_thickness - self.points['29'][0]
        self.stiffener_deflection = math.atan(deflection/stiffener_3_length)
            
    # convert pixels to millimeters
    def p2mm(self, value):
        return value / self.res_mm
    
    def forward(self):
        self.culc_hw()
        self.culc_line_thickness()
        self.culc_gaps()
        self.culc_tail()
        self.culc_stiffener()
        params = [self.height, self.width, self.top_line_thickness, 
                  self.bottom_line_thickness, self.gap_1, self.gap_2, 
                  self.tail_height, self.stiffener_deflection
                 ]
        for i in range(len(params)):
            round(params[i], 1)
        return params
                
    def writer(self, params, file_name):
        head = [
            ['N', 'Имя', 'Размер, мм', 'Норма, мм', 'Отклонение, мм', 'Отклонение, %', 'Допуск отклонения, мм', 'Статус'],
        ]
        
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            for row in head:
                writer.writerow(row)
        data = []
        N = 1
        for size, default in zip(params[:-1], self.default_params.values()):
            size = round(size, 1)
            name = default[2]
            norm = default[0]
            dif = round(size - norm, 1)
            difc = round(100 * abs(dif) / norm, 1)
            norm_dif = default[1]
            if abs(dif) <= norm_dif:
                status = 'Ок'
            else:
                status = 'Брак'
            data.append([N, name, size, norm, dif, difc, norm_dif, status])
            N += 1
        
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
    
    def draw(self, image_name, params):
        image = cv2.imread(image_name)
               
        # 'height': 24-1,
        self.draw_marks(image, params[0], self.points['24'], self.points['1'], 'left', 4)
        
        # 'width': 13-24,
        self.draw_marks(image, params[1], self.points['24'], self.points['13'], 'down', 4)
                            
        # 'top_line_thickness': 2-1,
        self.draw_marks(image, params[2], self.points['2'], self.points['1'], 'left', 1)
        
        # 'bottom_line_thickness': 24-23,
        self.draw_marks(image, params[3], self.points['24'], self.points['23'], 'left', 1)
        
        # 'gap_1': 21-22,
        self.draw_marks(image, params[4], self.points['21'], self.points['22'], 'down', 1)
        
        # 'gap_2': 17-18,
        self.draw_marks(image, params[5], self.points['17'], self.points['18'], 'down', 1)
        
        # 'tail_height': 13-11,
        self.draw_marks(image, params[6], self.points['13'], self.points['11'], 'right', 1)
        
        # 'stiffener_deflection': 29-5 
        
        cv2.imwrite(image_name, image)
        return image
            
    def draw_marks(self, image, param, point1, point2, side, k, color=(255,0,0), thickness=2):
        """
        side (str): left, right, top, down
        k (int): length koefficient
        """
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        
        if side == 'left':
            if point1[1] > point2[1]:
                point1, point2 = point2, point1
            image = cv2.line(image, point1, (point1[0]-k*20, point1[1]), color, thickness)    
            image = cv2.line(image, point2, (point2[0]-k*20, point2[1]), color, thickness) 
            image = cv2.line(image, (point1[0]-k*20, point1[1]-10), (point1[0]-k*20, point2[1]+10), color, thickness)
            text = str(round(param, 1))
            x, y = point1[0]-(k+2)*20, point2[1]-40
            image = cv2.putText(image, text, (x, y), font, fontScale, (255,255,0), thickness, cv2.LINE_AA)
            
        elif side == 'right':
            if point1[1] > point2[1]:
                point1, point2 = point2, point1
            image = cv2.line(image, point1, (point1[0]+k*20, point1[1]), color, thickness)    
            image = cv2.line(image, point2, (point2[0]+k*20, point2[1]), color, thickness) 
            image = cv2.line(image, (point1[0]+k*20, point1[1]-10), (point1[0]+k*20, point2[1]+10), color, thickness)
            text = str(round(param, 1))
            x, y = point1[0]+(k+2)*20, point2[1]-40
            image = cv2.putText(image, text, (x, y), font, fontScale, (255,255,0), thickness, cv2.LINE_AA)
            
        elif side == 'top':
            if point1[0] > point2[0]:
                point1, point2 = point2, point1
            image = cv2.line(image, point1, (point1[0], point1[1]-k*20), color, thickness)    
            image = cv2.line(image, point2, (point2[0], point2[1]-k*20), color, thickness) 
            image = cv2.line(image, (point1[0]-10, point1[1]-k*20), (point2[0]+10, point1[1]-k*20), color, thickness)
            text = str(round(param, 1))
            x, y = point1[0]+40, point2[1]-(k+2)*20
            image = cv2.putText(image, text, (x, y), font, fontScale, (255,255,0), thickness, cv2.LINE_AA)
            
        elif side == 'down':
            if point1[0] > point2[0]:
                point1, point2 = point2, point1
            image = cv2.line(image, point1, (point1[0], point1[1]+k*20), color, thickness)    
            image = cv2.line(image, point2, (point2[0], point2[1]+k*20), color, thickness) 
            image = cv2.line(image, (point1[0]-10, point1[1]+k*20), (point2[0]+10, point1[1]+k*20), color, thickness)
            text = str(round(param, 1))
            x, y = point1[0]+40, point2[1]+(k+2)*20
            image = cv2.putText(image, text, (x, y), font, fontScale, (255,255,0), thickness, cv2.LINE_AA)
        
        return image