#-*- coding: UTF-8 -*-
import cv2
import time
import threading
from .utils import *

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from capella_ros_msg.msg import CrowdStatistics
from rclpy.qos import qos_profile_system_default
import argparse
import os
import csv

import sys
from cuda import cuda
import numpy as np
from ctypes import *

import torch

from .nanodet.data.batch_process import stack_batch_img
from .nanodet.data.collate import naive_collate
from .nanodet.data.transform import Pipeline
from .nanodet.model.arch import build_model
from .nanodet.util import Logger, cfg, load_config, load_model_weight
from .Insightface_pytorch.inference import Match_Face

class EventData(Structure):
    _fields_ = [('client_id', c_int32),
                ('type', c_int32),
                ('user_data', c_void_p),
                ('data', c_void_p)]
    
class ImageData(Structure):
    _fields_ = [('image_id', c_int32),
                ('format', c_int32),
                ('width', c_int32),
                ('height', c_int32),
                ('pitch', c_int32),
                ('devptr', c_void_p)]
    

class Crowd_Statistics(Node):
        def __init__(self):
                super().__init__('crowd_statistics')
        
                self.viewer_publisher = self.create_publisher(CrowdStatistics,'ad_viewer_count',qos_profile_system_default)
                # self.color_subscription  # prevent unused variable warning

                # 创建行人定时筛选器，不断筛选已经没有观看的行人
                # self.select_timer = self.create_timer(1.0,self.pub_viewer_data)
                parser = argparse.ArgumentParser()
                parser.add_argument("--config", default=str(os.path.join(os.path.dirname(__file__), "model/nanodet-plus-m_416_person_face_817.yml")),
                                help="model config file path")
                parser.add_argument("--model", default=str(os.path.join(os.path.dirname(__file__), "model/nanodet-plus-m_416_person_face_817.pth")),
                                help="model file path")
                parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
                self.args = parser.parse_args(args=[])

                local_rank = 0
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

                load_config(cfg, self.args.config)
                self.logger = Logger(local_rank, use_tensorboard=False)
                self.device="cuda:0" if torch.cuda.is_available() else 'cpu'
                self.get_logger().info(self.device)
                self.cfg = cfg
                # 初始化nanodet模型
                model = build_model(cfg.model)
                ckpt = torch.load(self.args.model, map_location=lambda storage, loc: storage)
                load_model_weight(model, ckpt, self.logger)
                if cfg.model.arch.backbone.name == "RepVGG":
                        deploy_config = cfg.model
                        deploy_config.arch.backbone.update({"deploy": True})
                        deploy_model = build_model(deploy_config)
                        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

                        model = repvgg_det_model_convert(model, deploy_model)
                self.model = model.to(self.device).eval()
                self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
                self.get_logger().info('nanodet模型初始化完成')
                
                # 初始化arcface模型
                self.face_ = Match_Face(device=self.device)
                self.get_logger().info('arcface模型初始化完成')
                
                # 保存当前正在观看的行人数据
                self.track_dict = {}

                self.cuda_image = None
                self.is_save = False

                # update_image_thread = threading.Thread(target=self.update_image)
                # update_image_thread.start()
                self.update_image()


        def event_cb(self,data):
                if data[0].type == 2: # IMAGE_UPDATED
                        image = cast(data[0].data, self.PImageData)[0]
                        # print('image image_id:',image.image_id)
                        # print('image format:', image.format)
                        # print('image width:', image.width)
                        # print('image height:', image.height)
                        # print('image pitch:', image.pitch)
                        # print('image devptr:', image.devptr)

                        cpy = cuda.CUDA_MEMCPY2D()
                        cpy.WidthInBytes = 1080 * 4
                        cpy.Height = 1920
                        cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
                        cpy.srcDevice = image.devptr
                        cpy.srcXInBytes = 0
                        cpy.srcY = 0
                        cpy.srcPitch = image.pitch
                        cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_HOST
                        cpy.dstHost = self.cpuMat.__array_interface__['data'][0]
                        cpy.dstXInBytes = 0
                        cpy.dstY = 0
                        ret = cuda.cuMemcpy2D(cpy)
                        self.cuda_image = cv2.cvtColor(cv2.resize(self.cpuMat, (360, 640)), cv2.COLOR_RGBA2BGR)
                        
                        self.run(self.cuda_image,image.image_id)
                        # cv2.imshow('CUDA CAM {}'.format(1), cv2.cvtColor(cv2.resize(self.cpuMat, (360, 640)), cv2.COLOR_RGBA2BGR))
                        # cv2.waitKey(1)
                else:
                        self.cuda_image = None
                        self.get_logger().info('cuda image not update')


        def update_image(self,):             
                self.PEventData = POINTER(EventData)
                self.PImageData = POINTER(ImageData)
                self.EventCallback = CFUNCTYPE(None, self.PEventData)
                # cv2.cuda.createGpuMatFromCudaMemory().copyTo(cv2.cuda.GpuMat())

                self.cpuMat = np.zeros([1920, 1080, 4], np.uint8)

                print(self.cpuMat.__array_interface__['shape'])
                # cudash = cdll.LoadLibrary("./libcudash/libcudash.so")

                cudash = CDLL("/workspaces/capella_ros_docker/build/crowd_statistics/crowd_statistics/libcudash/libcudash.so")

                client_id=c_int32()
                cudash.cudash_create_client(byref(client_id), b"localhost", 6533, None, True)
                print("Client id: {}".format(client_id.value))

                cb_func = self.EventCallback(self.event_cb)
                cudash.cudash_client_set_event_callback(client_id, cb_func, None)
                cudash.cudash_open_image(client_id, c_int32(0))
                cudash.cudash_open_image(client_id, c_int32(1))

                input('Press ENTER to exit.')

                cudash.cudash_destroy_client(client_id)
                

        # 进行人脸检测和人脸匹配，将结果放到track_dict字典中
        def run(self, image, camera_id):
                self.pub_viewer_data()
                 # 计时
                start = time.perf_counter()
                # 读取图像
                # image = cv2.imread(r'/workspaces/crowd_statistics/crowd_statistics/555.jpg')
                
                origin_frame = image.copy()
                
                # 预处理图片，得到图片预处理后的数据，只要数据，发送到服务端的还是原图
                t1 = time.time()
                # nanodet获取人脸框
                meta, res = self.inference(image)
                t2 = time.time()
                # all_boxs：label, x0, y0, x1, y1, score
                result_frame,result_boxes = self.visualize(res[0], meta, cfg.class_names, 0.7)
                
                # arcface进行人脸识别
                if len(result_boxes) > 0:
                        # 筛选出脸部检测框
                        face_box = result_boxes[result_boxes[:,0] == 1]
                        for box in face_box:
                                box = box[1:-1]
                                # # 将脸部的框扩大1.5倍，增加脸部的特征
                                # h,w = int(box[3] - box[1]),int(box[2] - box[0])
                                # # 宽增加0.5倍，左右各增加0.25倍
                                # w_add = w/4
                                # h_add = h/4
                                # # 使用原始框的1.5倍进行预测
                                # face_frame = origin_frame[max(int(box[1] - h_add),0):min(int(box[3] + h_add),origin_frame.shape[0]-1), max(int(box[0] - w_add),0):min(int(box[2] + w_add),origin_frame.shape[1]-1)]
                                # 直接使用原始框进行预测
                                face_frame = origin_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                                cv2.imshow('new_face',face_frame)
                                cv2.waitKey(1)
                                # 人脸匹配
                                min_idx, minimum, face_uuid = self.face_.get_face_feature(face_frame,True)
                                if min_idx != -1:
                                        current_time = time.time()
                                        cv2.putText(result_frame,f'name:{face_uuid}',(int(box[0]),int(box[3])+15),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1)
                                        if face_uuid not in self.track_dict.keys():
                                                # 新增加一个脸
                                                self.track_dict[face_uuid] = {'start_time':current_time,'end_time':current_time,'camera_id':camera_id,'is_pub':False}
                
                                        else:
                                                # 如果两秒之类再次检测到就认为一直在观看
                                                if current_time - self.track_dict[face_uuid]['end_time'] <= 2:
                                                        self.track_dict[face_uuid]['end_time'] = current_time
                                                                 


                # a =  cv2.imwrite(r'/workspaces/crowd_statistics/123.jpg',result_frame)
                # print(a)
                cv2.namedWindow('face_match',0)
                cv2.imshow('face_match',result_frame)
                cv2.waitKey(1)
                

        def inference(self, img):
                img_info = {"id": 0}
                if isinstance(img, str):
                        img_info["file_name"] = os.path.basename(img)
                        img = cv2.imread(img)
                else:
                        img_info["file_name"] = None

                height, width = img.shape[:2]
                img_info["height"] = height
                img_info["width"] = width
                meta = dict(img_info=img_info, raw_img=img, img=img)
                meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
                meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
                meta = naive_collate([meta])
                meta["img"] = stack_batch_img(meta["img"], divisible=32)
                with torch.no_grad():
                        results = self.model.inference(meta)
                return meta, results
        

        def visualize(self, dets, meta, class_names, score_thres, wait=0):
                time1 = time.time()
                result_img,all_box = self.model.head.show_result(
                meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
                )
                # self.get_logger().info("viz time: {:.3f}s".format(time.time() - time1))
                return result_img,np.array(all_box)
        
        
        def pub_viewer_data(self,):
                # 每一次都会检测track_dict是否有已经没在观看屏幕的人了，然后删除
                dict_keys = list(self.track_dict.keys())
                for key in dict_keys:
                        current_time = time.time()
                        # fa bu kai shi guan kan shu ju 
                        if self.track_dict[key]['end_time'] - self.track_dict[key]['start_time'] > 0.5 and self.track_dict[key]['is_pub'] != True:
                                viewer_msg = CrowdStatistics()
                                viewer_msg.camera_id = self.track_dict[key]['camera_id']
                                viewer_msg.face_uuid = key
                                viewer_msg.start_end = True
                                self.viewer_publisher.publish(viewer_msg)
                                self.track_dict[key]['is_pub'] = True


                        if current_time - self.track_dict[key]['end_time'] > 2:
                                if self.track_dict[key]['is_pub'] == True:
                                        viewer_msg = CrowdStatistics()
                                        viewer_msg.camera_id = self.track_dict[key]['camera_id']
                                        viewer_msg.face_uuid = key
                                        viewer_msg.start_end = False
                                        self.viewer_publisher.publish(viewer_msg)
                                        # print(self.track_dict[key]['end_time'] - self.track_dict[key]['start_time'])
                                        if self.is_save:
                                                # 是否行人观看记录文件
                                                csv_save_path = os.path.join(os.path.dirname(__file__),r'Insightface_pytorch/work_space/{}_watching_records.csv'.format(time.strftime("%Y_%m_%d", time.localtime())))
                                                if not os.path.exists(csv_save_path):
                                                        with open(csv_save_path, 'w', newline='', encoding='utf-8') as csvfile:
                                                                csv_writer = csv.writer(csvfile)
                                                                csv_writer.writerow(['id', 'start_time', 'end_time', 'duration'])
                                                                csv_writer.writerow([key,self.track_dict[key]['start_time'],self.track_dict[key]['end_time'],self.track_dict[key]['end_time'] - self.track_dict[key]['start_time']])
                                                else:
                                                        with open(csv_save_path, 'a', newline='') as csvfile:
                                                                csv_writer = csv.writer(csvfile)
                                                                csv_writer.writerow([key,self.track_dict[key]['start_time'],self.track_dict[key]['end_time'],self.track_dict[key]['end_time'] - self.track_dict[key]['start_time']])
                                                self.get_logger().info(f"{key}观看了：{self.track_dict[key]['end_time'] - self.track_dict[key]['start_time']} s")
                                                self.get_logger().info(r'人脸数据文件保存路径为：{}'.format(os.path.dirname(csv_save_path)))
                                del self.track_dict[key]




def main(args=None):
        rclpy.init(args=args)

        minimal_subscriber = Crowd_Statistics()

        rclpy.spin(minimal_subscriber)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
        main()