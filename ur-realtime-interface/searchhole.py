#!/usr/bin/python3
import URBasic
import time
import math
import NetFT
import numpy as np
import csv
s2tcptran = np.array([[0.5179, -0.8959, 0], [0.8959, 0.5179, 0],
                      [0, 0, 1]])  # sensor to tcp rotation matrix,expressed in sensor frame
G = np.array([0, 0, -11.1431])  # tool gravity,expressed in base frame
p = np.array([[0, -0.0545,  0.0034], [0.0545, 0, 0.0023],
              [- 0.0034, -0.0023, 0]])  # mass center in sensor frame
# sensor initial offset
fd = np.array([-0.3819, -1.3764, -8.9912,  -0.0131,    -0.0841,      -0.0328])
tcp2sensor = np.linalg.inv(s2tcptran)
zero = np.zeros((3, 3), dtype=float)
vcc = 0.1
acc = 0.1


class PIDController:
    def __init__(self):
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host='192.168.1.50', robotModel=URBasic.robotModel.RobotModel())
        # print(self.robot.get_actual_tcp_pose()[3:6])
        self.sensor = NetFT.Sensor('192.168.1.30')
        self.P = np.diag((0.00, 0.00, 0.0001, 0.00, 0.00, 0.00))
        self.I = np.diag((0, 0, 0, 0, 0, 0))
        self.D = np.diag((0, 0, 0.0001, 0, 0, 0))
        self.v = [0, 0, 0, 0, 0, 0]
        self.poses = [0, 0, 0, 0, 0, 0]
        self.target_ft = [0, 0, 5, 0, 0, 0]

    def AxisAng2RotaMatri(self, angle_vec):
        '''
        Convert an Axis angle to rotation matrix
        AxisAng2Matrix(angle_vec)for i in range(3):
                if abs(self.v[i]) < 0.05 or abs(self.v[i]) > 1.5:
                    self.v[i] = 0
            for i in range(3, 6):
                if abs(self.v[i]) < 0.01 or abs(self.v[i]) > 1.0:
                    self.v[i] = 0
        angle_vec need to be a 3D Axis angle
        '''
        theta = math.sqrt(angle_vec[0]**2+angle_vec[1]**2+angle_vec[2]**2)
        if theta == 0.:
            return np.identity(3, dtype=float)

        cs = np.cos(theta)
        si = np.sin(theta)
        e1 = angle_vec[0]/theta
        e2 = angle_vec[1]/theta
        e3 = angle_vec[2]/theta

        R = np.zeros((3, 3))
        R[0, 0] = (1-cs)*e1**2+cs
        R[0, 1] = (1-cs)*e1*e2-e3*si
        R[0, 2] = (1-cs)*e1*e3+e2*si
        R[1, 0] = (1-cs)*e1*e2+e3*si
        R[1, 1] = (1-cs)*e2**2+cs
        R[1, 2] = (1-cs)*e2*e3-e1*si
        R[2, 0] = (1-cs)*e1*e3-e2*si
        R[2, 1] = (1-cs)*e2*e3+e1*si
        R[2, 2] = (1-cs)*e3**2+cs
        return R

    def gravitycomp(self, ft, rotate_i):
        '''
         this is a gravit compensation function to compensate the tool gravity and obtain the precise
        force on the load.
        '''
        # axisangle = self.robot.get_target_tcp_poImpedancese()[3:6]
        # rotate = URBasic.kinematic.AxisAng2RotaMatri(axisangle)
        # tcp rotation matrix expressed in tcp frame
        trans = np.dot(s2tcptran, rotate_i)
        # in sensor frame
        Ftrans = np.dot(trans, G)
        Mtrans = np.dot(p, Ftrans)
        fttrans = np.hstack((Ftrans, Mtrans))
        ftcomp = ft-fttrans-fd
        # print('ok')
        return ftcomp

    def run(self):
        # count = 1
        x = [-0.15069097207881152, -0.39985470329679895, 0.4046209621844865, -
             2.4926138340994397, -1.911217721409695, -0.006238534241991061]
        self.robot.movel(pose=x, a=1.2, v=1.0)
        self.target_ft = np.reshape(self.target_ft, (6, 1))
        self.v = np.reshape(self.v, (6, 1))
        self.poses = np.reshape(self.poses, (6, 1))
        init_pose = self.robot.get_actual_tcp_pose()
        self.robot.movel(pose=init_pose.tolist(), a=1.2, v=1.0)
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        init_ft = self.gravitycomp(ft, np.linalg.inv(rotate))
        # z0 = init_pose[2]
        err_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        err_i = np.reshape(err_i, (6, 1))
        err_d = np.array([0, 0, 0, 0, 0, 0])
        err_d = np.reshape(err_d, (6, 1))
        traj = self.searchtraj()
        pose_base = np.reshape(np.array(init_pose), (6, 1))
        res = []
        for i in range(traj.shape[0]):
            ft = np.array(self.sensor.tare()) / 1000000.0
            # print(ft)
            pose = self.robot.get_actual_tcp_pose()

            axisangle = pose[-3:]
            rotate = self.AxisAng2RotaMatri(axisangle)
            ftcomp = self.gravitycomp(ft, np.linalg.inv(rotate))-init_ft

            base2sensor = np.dot(rotate, tcp2sensor)
            base2sensor1 = np.c_[base2sensor, zero]
            base2sensor2 = np.c_[zero, base2sensor]
            base2sensortran = np.r_[base2sensor1, base2sensor2]
            ft_base = np.reshape(np.dot(base2sensortran, ftcomp), (6, 1))
            ft_tmp = ft_base.tolist()
            ft_record = [i for item in ft_tmp for i in item]
            # print (ft_record)
            pose_record = pose.tolist()
            res.append(ft_record+pose_record)
            # print(ft_record+pose_record)
            err = self.target_ft-ft_base

            err_i = err_i + err
            err_l = err-err_d
            err_d = err
            if i > 0:
                self.v = -(np.dot(self.P, err)+np.dot(self.D, err_d)+np.dot(self.I, err_i))*0.1
            # print(self.v)
            # count += 1
            # use movel  s=v*t,t=1
            # self.v[2] = 0.0001
            self.poses = self.v+pose_base+np.reshape(traj[i], (6, 1))
            # for i in range(2):
            #     self.v[i] = vcc
            # for i in range(3):
            #     if abs(ft_base[i] < 0.5 or abs(ft_base[i]) > 50):
            #         self.v[i] = 0
            # for i in range(3, 6):
            #     if abs(ft_base[i] < 0.1 or abs(ft_base[i]) > 20):
            #         self.v[i] = 0
            # for i in range(3):
            #     if abs(self.v[i]) < 0.05 or abs(self.v[i]) > 1.5:
            #         self.v[i] = 0
            pose_tmp = self.poses.tolist()
            pose_cmd = [i for item in pose_tmp for i in item]
            # print(pose_cmd)
            self.robot.movel(pose=pose_cmd, a=acc, v=vcc)
            # time.sleep(0.008)

        self.robot.stopl()
        self.robot.close()
        with open("data.csv", 'w', newline='') as t:
            b = ["fx", "fy", "fz", "tx", "ty", "tz", "px", "py", "pz", "rx", "ry", "rz"]
            writer = csv.writer(t)
            writer.writerow(b)
            writer.writerows(res)

    def comptest(self):
        # init_ft = np.array(self.sensor.tare()) / 1000000.0
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        init_ft = self.gravitycomp(ft, np.linalg.inv(rotate))
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        ftcomp = self. gravitycomp(ft, np.linalg.inv(rotate))-init_ft
        print (ftcomp)

    def searchtraj(self):
        deltal = 0.0002
        rows = 101
        cols = 101
        # straj = np.zeros((rows, cols), dtype=float)
        traj = np.zeros((rows*cols, 6), dtype=float)
        for i in range(rows):
            for j in range(cols):
                traj[i*cols+j][0] = (j-(rows-1)/2)*deltal
                traj[i*cols+j][1] = (i-(rows-1)/2)*deltal
        return traj


if __name__ == "__main__":
    controller = PIDController()
    # controller.run()  # flag=true,peg-inhole else teach_mode
    controller.comptest()
    # controller.searchtraj()
